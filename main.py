import os

import requests
import trafilatura
from dotenv import load_dotenv
from google import genai
from google.genai import types
from googlenewsdecoder import gnewsdecoder
from pygooglenews import GoogleNews

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}


# Step 1. Google News에서 카테고리별 후보 기사 30개를 수집합니다.
def fetch_top_news():
    print("[Step 1] 글로벌 구글 뉴스 데이터 수집...")

    seen_links = set()

    if os.path.exists("history.txt"):
        with open("history.txt", "r", encoding="utf-8") as f:
            seen_links.update(line.strip() for line in f.readlines())
        print(f"기록된 뉴스 {len(seen_links)}건을 블랙리스트에 선탑재했습니다.")
    else:
        print("기록된 뉴스가 없습니다.")

    gn_kr = GoogleNews(lang="ko", country="KR")
    gn_us = GoogleNews(lang="en", country="US")
    raw_news = []

    def add_news(entries, category_name):
        added_count = 0

        for entry in entries:
            if entry.link in seen_links:
                continue

            source = ""
            if hasattr(entry, "source") and entry.source:
                source = entry.source.get("title", "")

            raw_news.append(
                {
                    "id": len(raw_news) + 1,
                    "category": category_name,
                    "title": entry.title,
                    "source": source,
                    "google_link": entry.link,
                }
            )

            seen_links.add(entry.link)
            added_count += 1

            if added_count >= 10:
                break

    try:
        kr_top = gn_kr.top_news()
        print(" -> 한국 종합 헤드라인 수집 완료")
        add_news(kr_top["entries"], "종합(KR)")
    except Exception as e:
        print(f"한국 종합 뉴스 수집 실패: {e}")

    try:
        kr_biz = gn_kr.topic_headlines("BUSINESS")
        print(" -> 한국 경제 헤드라인 수집 완료")
        add_news(kr_biz["entries"], "경제(KR)")
    except Exception as e:
        print(f"한국 경제 뉴스 수집 실패: {e}")

    try:
        us_biz = gn_us.topic_headlines("BUSINESS")
        print(" -> 미국 경제 헤드라인 수집 완료")
        add_news(us_biz["entries"], "경제(US)")
    except Exception as e:
        print(f"미국 경제 뉴스 수집 실패: {e}")

    return raw_news


# Step 2. Gemini에게 전달할 기사 후보 목록을 만듭니다. 링크는 보내지 않습니다.
def build_news_context(news_list):
    lines = []

    for news in news_list:
        lines.append(
            "\n".join(
                [
                    f"ID: {news['id']}",
                    f"Category: {news['category']}",
                    f"Title: {news['title']}",
                    f"Source: {news['source']}",
                ]
            )
        )

    return "\n\n".join(lines)


# Step 3. Gemini가 카테고리별 Best 기사 ID를 선정합니다.
def select_best_articles(news_list):
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(".env 파일에 GEMINI_API_KEY를 먼저 입력하세요.")

    news_context = build_news_context(news_list)

    prompt = f"""**Role:** Senior Strategic News Analyst & Professional Curator.

**Objective:** From the provided list of 30 news articles, identify and select the single most impactful "Best" article from each category. Your goal is to provide high-value intelligence that a COO would find indispensable.

**Strict Selection Criteria (Priority-based):**
1. [종합(KR)]: Choose the article with the highest social urgency or national importance. Prioritize breaking news that affects the general public.
2. [경제(KR)]: Choose the article that signals a major shift in the Korean market. Prioritize macro-economic data (interest rates, inflation) or game-changing moves by top-tier conglomerates (Samsung, SK, Hyundai, etc.).
3. [경제(US)]: Choose the article with global repercussions. Prioritize Federal Reserve policy shifts, AI/Big Tech disruptions, or critical changes in the global supply chain.

**Selection Logic:**
- If multiple articles meet the criteria, select the one that is most "actionable" or "insightful" for business strategy.
- Return only the selected article ID for each category.
- Do not return title, source, link, summary, or commentary.
- You must select one ID from each category: 종합(KR), 경제(KR), 경제(US).

**Output Format (Strictly for machine parsing):**
Category: [Category Name]
Selected ID: [Article ID]

Category: [Category Name]
Selected ID: [Article ID]

Category: [Category Name]
Selected ID: [Article ID]

---
**News List to Analyze:**
{news_context}"""

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )

    return response.text.strip()


# Step 4-1. Gemini가 반환한 Selected ID를 파싱합니다.
def parse_selected_ids(selected_result):
    selected_items = []
    current_item = {}

    for line in selected_result.splitlines():
        line = line.strip()

        if line.startswith("Category:"):
            if current_item:
                selected_items.append(current_item)
                current_item = {}
            current_item["category"] = line.replace("Category:", "").strip()

        elif line.startswith("Selected ID:"):
            selected_id_text = line.replace("Selected ID:", "").strip()
            current_item["selected_id"] = int(selected_id_text)

    if current_item:
        selected_items.append(current_item)

    return selected_items


# Step 4-2. Selected ID로 원본 기사 데이터를 찾습니다.
def match_selected_articles(selected_result, news_list):
    selected_items = parse_selected_ids(selected_result)
    news_by_id = {news["id"]: news for news in news_list}

    selected_articles = []

    for item in selected_items:
        selected_id = item["selected_id"]
        article = news_by_id.get(selected_id)

        if not article:
            print(f"선택된 ID를 찾을 수 없습니다: {selected_id}")
            continue

        selected_articles.append(article)

    return selected_articles


# Step 5-1. Google News 암호화 링크를 실제 언론사 URL로 변환합니다.
def resolve_article_url(google_link):
    try:
        decoded_result = gnewsdecoder(google_link, interval=1)

        if decoded_result.get("status"):
            resolved_link = decoded_result["decoded_url"]
            print(f" -> 원문 URL: {resolved_link}")
            return resolved_link

        print(f"URL 정화 실패: {decoded_result.get('message')}")
        return ""

    except Exception as e:
        print(f"URL 정화 중 오류 발생: {e}")
        return ""


# Step 5-2. 선택된 3개 기사 링크를 정화합니다.
def resolve_selected_article_links(selected_articles):
    for article in selected_articles:
        print(f"URL 정화 중: {article['title'][:30]}...")
        article["resolved_link"] = resolve_article_url(article["google_link"])

    return selected_articles


# Step 6-1. 실제 언론사 URL에서 기사 본문을 추출합니다.
def fetch_article_body(resolved_link):
    try:
        response = requests.get(
            resolved_link,
            headers=REQUEST_HEADERS,
            timeout=20,
            allow_redirects=True,
        )
        response.raise_for_status()

    except requests.RequestException as e:
        print(f"본문 페이지 다운로드 실패: {resolved_link} ({e})")
        return "", "download_failed"

    body = trafilatura.extract(
        response.text,
        url=resolved_link,
        include_comments=False,
        include_tables=False,
    )

    body = body or ""

    if len(body.strip()) < 300:
        print(" -> 본문 추출 실패 또는 본문이 너무 짧습니다.")
        return body, "extract_failed"

    print(f" -> 본문 추출 완료: {len(body.strip())}자")
    return body, "success"


# Step 6-2. 선택된 3개 기사 본문을 수집합니다.
def fetch_selected_article_bodies(selected_articles):
    for article in selected_articles:
        print(f"본문 수집 중: {article['title'][:30]}...")

        if not article.get("resolved_link"):
            article["body"] = ""
            article["status"] = "resolve_failed"
            print(" -> 원문 URL이 없어 본문 수집을 건너뜁니다.")
            continue

        article["body"], article["status"] = fetch_article_body(
            article["resolved_link"]
        )

    return selected_articles


# Step 7-1. 인스타 캡션과 Flux 이미지 프롬프트 생성을 위한 Gemini 프롬프트를 만듭니다.
def build_instagram_prompt(article):
    return f"""[Persona]
You are a top-tier Instagram News Curator and Visual Director. Your mission is to transform raw news into a viral, professional Instagram post. You think like a human editor—friendly, trendy, and factual.

[Input Data]
- Title: {article['title']}
- Source: {article['source']}
- Body: {article['body']}

[Task 1: Instagram Caption (KOREAN ONLY)]
Write a high-engagement Instagram caption in KOREAN based on the input data.
1. Headline: Start with "[속보🚨]" followed by a punchy headline.
2. Hook: A catchy opening sentence to stop the scroll.
3. Summary: 3 clear bullet points using 📍 or ✅. (Facts only, no hallucinations).
4. Context: A friendly explanation of why this news matters.
5. Tone: NEVER use AI-typical phrases like "결론적으로", "요약하자면", "따라서", "알아보겠습니다". Use natural K-Instagram endings like "~하네요!", "~입니다", or "대박이죠?".
6. Source: Write "출처: {article['source']}" at the end.
7. Hashtags: Exactly 4 relevant hashtags.

[Task 2: Flux.1 Image Prompt (ENGLISH ONLY)]
Create a professional image generation prompt for Flux.1 to create a symbolic editorial visual for the news topic.
1. Style: Photojournalism, editorial photography, candid shot, shot on 35mm lens, high-fidelity, raw photo textures.
2. Composition: 4:5 Portrait aspect ratio. Ensure the main subject is in the upper 3/4 of the frame.
3. Negative Space: Include "significant negative space at the bottom 1/4 of the frame".
4. Mandatory Gradient: Explicitly state "Apply a soft, dark vertical gradient fade to black at the bottom 1/4 of the image for text legibility".
5. Local Context: If the news is Korean, describe a "Modern Seoul background" or "Korean context".
6. Constraints: No text in the image. No artificial glow. No over-saturation.
7. Safety: Do not depict identifiable real people or recreate exact faces. Use symbolic or generic subjects when needed.
8. Interpretation: Represent the topic visually without claiming to show the actual event.

[Output Format]
===PART 1: CAPTION===
(Your Korean Caption here)

===PART 2: IMAGE_PROMPT===
(Your English Flux Prompt here)"""


# Step 7-2. Gemini 응답에서 캡션과 Flux 프롬프트를 분리합니다.
def parse_instagram_generation(raw_text):
    caption = ""
    image_prompt = ""

    caption_marker = "===PART 1: CAPTION==="
    image_marker = "===PART 2: IMAGE_PROMPT==="

    if caption_marker in raw_text and image_marker in raw_text:
        caption_part = raw_text.split(caption_marker, 1)[1]
        caption, image_prompt = caption_part.split(image_marker, 1)
        caption = caption.strip()
        image_prompt = image_prompt.strip()
    else:
        caption = raw_text.strip()

    return caption, image_prompt


# Step 7-3. 기사 1개에 대해 인스타 캡션과 Flux 프롬프트를 생성합니다.
def generate_instagram_post(article):
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(".env 파일에 GEMINI_API_KEY를 먼저 입력하세요.")

    if article.get("status") != "success" or not article.get("body"):
        article["instagram_post_raw"] = ""
        article["instagram_caption"] = ""
        article["flux_prompt"] = ""
        article["instagram_post_status"] = "skipped_no_body"
        return article

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=build_instagram_prompt(article),
        config=types.GenerateContentConfig(temperature=0.7),
    )

    raw_text = response.text.strip()
    caption, image_prompt = parse_instagram_generation(raw_text)

    article["instagram_post_raw"] = raw_text
    article["instagram_caption"] = caption
    article["flux_prompt"] = image_prompt
    article["instagram_post_status"] = "success"

    return article


# Step 7-4. 선택된 3개 기사에 대해 인스타 콘텐츠를 생성합니다.
def generate_instagram_posts(selected_articles):
    for article in selected_articles:
        print(f"인스타 게시물 생성 중: {article['title'][:30]}...")
        generate_instagram_post(article)

    return selected_articles


# Step 8-1. 선택된 기사 메타데이터를 저장합니다.
def save_selected_news(selected_articles):
    with open("selected_news.txt", "w", encoding="utf-8") as f:
        for article in selected_articles:
            f.write(f"ID: {article['id']}\n")
            f.write(f"Category: {article['category']}\n")
            f.write(f"Title: {article['title']}\n")
            f.write(f"Source: {article['source']}\n")
            f.write(f"Google Link: {article['google_link']}\n")
            f.write(f"Resolved Link: {article.get('resolved_link', '')}\n")
            f.write(f"Status: {article.get('status', '')}\n")
            f.write(f"Instagram Post Status: {article.get('instagram_post_status', '')}\n")
            f.write("\n---\n\n")


# Step 8-2. 본문, 인스타 캡션, Flux 프롬프트까지 저장합니다.
def save_selected_articles(selected_articles):
    with open("selected_articles.txt", "w", encoding="utf-8") as f:
        for article in selected_articles:
            f.write(f"ID: {article['id']}\n")
            f.write(f"Category: {article['category']}\n")
            f.write(f"Title: {article['title']}\n")
            f.write(f"Source: {article['source']}\n")
            f.write(f"Google Link: {article['google_link']}\n")
            f.write(f"Resolved Link: {article.get('resolved_link', '')}\n")
            f.write(f"Status: {article.get('status', '')}\n")
            f.write(f"Instagram Post Status: {article.get('instagram_post_status', '')}\n")
            f.write("Body:\n")
            f.write(article.get("body", ""))
            f.write("\n\nInstagram Caption:\n")
            f.write(article.get("instagram_caption", ""))
            f.write("\n\nFlux Prompt:\n")
            f.write(article.get("flux_prompt", ""))
            f.write("\n\n---\n\n")


# Step 9. 전체 파이프라인을 실행합니다.
if __name__ == "__main__":
    news_list = fetch_top_news()

    for news in news_list[:3]:
        print(news)

    print(f"\n필터링 완료. 총 {len(news_list)}개의 신선한 뉴스를 확보했습니다.\n")

    if len(news_list) > 0:
        print("--- [Gemini 선정 결과] ---")
        selected_result = select_best_articles(news_list)
        print(selected_result)

        with open("gemini_selected_result.txt", "w", encoding="utf-8") as f:
            f.write(selected_result)

        selected_articles = match_selected_articles(selected_result, news_list)
        selected_articles = resolve_selected_article_links(selected_articles)
        selected_articles = fetch_selected_article_bodies(selected_articles)
        selected_articles = generate_instagram_posts(selected_articles)

        save_selected_news(selected_articles)
        save_selected_articles(selected_articles)

        with open("history.txt", "w", encoding="utf-8") as f:
            f.write(news_list[0]["google_link"] + "\n")

        print(
            f"\n[테스트 알림] '{news_list[0]['title'][:20]}...' 기사를 history.txt에 기록했습니다."
        )
        print("한 번 더 실행하면 위 기사가 블랙리스트에 의해 걸러지는지 확인하세요!")
    else:
        print("수집된 뉴스가 없습니다. 구글 뉴스 연결 상태를 확인하세요.")
