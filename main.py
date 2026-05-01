import os
import requests
import trafilatura
import json
import shutil
from dotenv import load_dotenv
from google import genai
from google.genai import types
from googlenewsdecoder import gnewsdecoder
from pygooglenews import GoogleNews
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import InferenceClient

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

def create_run_dir():
    today = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path("outputs") / today
    image_dir = run_dir / "images"

    image_dir.mkdir(parents=True, exist_ok=True)

    return run_dir

# history.jsonl에서 이미 사용한 기사 링크를 읽어옵니다.
def load_seen_links():
    seen_links = set()
    history_path = Path("history.jsonl")

    if not history_path.exists():
        print("기록된 뉴스가 없습니다.")
        return seen_links

    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            google_link = record.get("google_link")
            if google_link:
                seen_links.add(google_link)

    print(f"기록된 뉴스 {len(seen_links)}건을 블랙리스트에 선탑재했습니다.")
    return seen_links

# Step 1. Google News에서 카테고리별 후보 기사 30개를 수집합니다.
def fetch_top_news():
    print("[Step 1] 글로벌 구글 뉴스 데이터 수집...")
    
    seen_links = load_seen_links()
    
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
- Return exactly two selected article IDs for each category.
- The first ID is the primary choice.
- The second ID is the backup choice if the primary article fails during processing.
- Do not return title, source, link, summary, or commentary.
- You must select two IDs from each category: 종합(KR), 경제(KR), 경제(US).

**Output Format (Strictly for machine parsing):**
Category: [Category Name]
Primary ID: [Article ID]
Backup ID: [Article ID]

Category: [Category Name]
Primary ID: [Article ID]
Backup ID: [Article ID]

Category: [Category Name]
Primary ID: [Article ID]
Backup ID: [Article ID]

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

# Step 4-1. Gemini가 반환한 Primary ID와 Backup ID를 파싱합니다.
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

        elif line.startswith("Primary ID:"):
            primary_id_text = line.replace("Primary ID:", "").strip()
            current_item["primary_id"] = int(primary_id_text)

        elif line.startswith("Backup ID:"):
            backup_id_text = line.replace("Backup ID:", "").strip()
            current_item["backup_id"] = int(backup_id_text)

    if current_item:
        selected_items.append(current_item)

    return selected_items

# Step 4-2. Primary/Backup ID로 원본 기사 데이터를 찾습니다.
def match_selected_articles(selected_result, news_list):
    selected_items = parse_selected_ids(selected_result)
    news_by_id = {news["id"]: news for news in news_list}

    selected_articles = []

    for item in selected_items:
        category = item["category"]

        primary_article = news_by_id.get(item.get("primary_id"))
        backup_article = news_by_id.get(item.get("backup_id"))

        if primary_article:
            primary_article = primary_article.copy()
            primary_article["selection_rank"] = "primary"
            primary_article["backup_article"] = backup_article.copy() if backup_article else None
            selected_articles.append(primary_article)
        else:
            print(f"1순위 ID를 찾을 수 없습니다: {item.get('primary_id')}")

            if backup_article:
                backup_article = backup_article.copy()
                backup_article["selection_rank"] = "backup"
                backup_article["backup_article"] = None
                selected_articles.append(backup_article)
            else:
                print(f"2순위 ID도 찾을 수 없습니다: {category}")

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

# Step 7-1. 인스타 캡션 생성을 위한 Gemini 프롬프트를 만듭니다.
def build_instagram_caption_prompt(article):
    return f"""[Persona]
You are a professional Instagram News Curator. Your goal is to rewrite complex news into a viral, human-centric post.

[Input Data]
- Title: {article['title']}
- Source: {article['source']}
- Body: {article['body']}

[Task: Instagram Caption (KOREAN ONLY)]
Write a high-engagement Instagram caption based on the input data.
1. Headline: Start with "[속보🚨]" followed by a punchy, click-worthy headline.
2. Hook: A catchy opening sentence to stop the scroll.
3. Summary: 3 clear, punchy bullet points using 📍 or ✅. (Facts only, no hallucinations).
4. Context: A friendly explanation of why this news is important to the reader.
5. Tone: Use natural K-Instagram endings like "~하네요!", "~입니다", or "대박이죠?". Strictly avoid "AI-ish" connectors like "따라서", "결론적으로".
6. Source: "출처: {article['source']}"
7. Hashtags: Exactly 4 relevant hashtags.

[Output Format]
===KOREAN_CAPTION===
(Your caption here)"""


def parse_instagram_caption(raw_text):
    marker = "===KOREAN_CAPTION==="

    if marker in raw_text:
        return raw_text.split(marker, 1)[1].strip()

    return raw_text.strip()


# Step 7-2. 기사 1개에 대해 인스타 캡션을 생성합니다.
def generate_instagram_caption(article):
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(".env 파일에 GEMINI_API_KEY를 먼저 입력하세요.")

    if article.get("status") != "success" or not article.get("body"):
        article["instagram_caption_raw"] = ""
        article["instagram_caption"] = ""
        article["instagram_caption_status"] = "skipped_no_body"
        return article

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=build_instagram_caption_prompt(article),
        config=types.GenerateContentConfig(temperature=0.7),
    )

    raw_text = response.text.strip()

    article["instagram_caption_raw"] = raw_text
    article["instagram_caption"] = parse_instagram_caption(raw_text)
    article["instagram_caption_status"] = "success"

    return article


# Step 7-3. 선택된 기사들에 대해 인스타 캡션을 생성합니다.
def generate_instagram_captions(selected_articles):
    for article in selected_articles:
        print(f"인스타 캡션 생성 중: {article['title'][:30]}...")
        generate_instagram_caption(article)

    return selected_articles


def save_instagram_captions(selected_articles, run_dir):
    with open(run_dir / "instagram_captions.txt", "w", encoding="utf-8") as f:
        for article in selected_articles:
            f.write(f"ID: {article['id']}\n")
            f.write(f"Category: {article['category']}\n")
            f.write(f"Title: {article['title']}\n")
            f.write(f"Source: {article['source']}\n")
            f.write(f"Status: {article.get('instagram_caption_status', '')}\n")
            f.write("Instagram Caption:\n")
            f.write(article.get("instagram_caption", ""))
            f.write("\n\n---\n\n")


# Step 8-1. 인스타 캡션을 기반으로 SDXL 이미지 프롬프트를 만듭니다.
def build_sdxl_image_prompt(article):
    step1_output = article.get("instagram_caption", "")

    return f"""[Persona]
You are a Visual Director specializing in photojournalism. You transform text-based news summaries into highly optimized keyword-based prompts for Stable Diffusion XL (SDXL).

[Input Data]
- Generated Caption: {step1_output}

[Task: SDXL Image Prompt (ENGLISH ONLY, KEYWORD FORMAT)]
Analyze the caption and create a symbolic editorial visual prompt. You MUST output a comma-separated list of keywords, NOT full sentences.

1. Core Concept: Extract the most striking visual element (e.g., "A solitary figure in a dark office", "A glowing stock market chart").
2. Format Rules: Use this exact structure -> [Core Concept], [Style], [Composition], [Lighting/Atmosphere].
3. Style Keywords: Photojournalism, editorial photography, candid shot, shot on 35mm lens, 8k resolution, highly detailed, photorealistic.
4. Composition & Layout (CRITICAL):
   - Include keywords: "vertical portrait", "main subject in upper half".
   - Include keywords for text space: "dark negative space at bottom", "soft black gradient at bottom edge", "vignette".
5. Local Context: Add "Seoul street" or "Korean context" ONLY if relevant.
6. Constraints: Include "no text", "no watermarks" in the prompt. Do not depict identifiable real people. Use symbolic subjects.

[Output Format]
===IMAGE_PROMPT===
(Provide only the comma-separated English keywords here)"""


def parse_sdxl_image_prompt(raw_text):
    marker = "===IMAGE_PROMPT==="

    if marker in raw_text:
        return raw_text.split(marker, 1)[1].strip()

    return raw_text.strip()


# Step 8-2. 기사 1개에 대해 SDXL 이미지 프롬프트를 생성합니다.
def generate_sdxl_image_prompt(article):
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(".env 파일에 GEMINI_API_KEY를 먼저 입력하세요.")

    if not article.get("instagram_caption"):
        article["sdxl_image_prompt_raw"] = ""
        article["sdxl_image_prompt"] = ""
        article["sdxl_image_prompt_status"] = "skipped_no_caption"
        return article

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=build_sdxl_image_prompt(article),
        config=types.GenerateContentConfig(temperature=0.7),
    )

    raw_text = response.text.strip()

    article["sdxl_image_prompt_raw"] = raw_text
    article["sdxl_image_prompt"] = parse_sdxl_image_prompt(raw_text)
    article["sdxl_image_prompt_status"] = "success"

    return article


# Step 8-3. 선택된 기사들에 대해 SDXL 이미지 프롬프트를 생성합니다.
def generate_sdxl_image_prompts(selected_articles):
    for article in selected_articles:
        print(f"SDXL 이미지 프롬프트 생성 중: {article['title'][:30]}...")
        generate_sdxl_image_prompt(article)

    return selected_articles


def save_sdxl_image_prompts(selected_articles, run_dir):
    with open(run_dir / "sdxl_image_prompts.txt", "w", encoding="utf-8") as f:
        for article in selected_articles:
            f.write(f"ID: {article['id']}\n")
            f.write(f"Category: {article['category']}\n")
            f.write(f"Title: {article['title']}\n")
            f.write(f"Source: {article['source']}\n")
            f.write(f"Status: {article.get('sdxl_image_prompt_status', '')}\n")
            f.write("SDXL Image Prompt:\n")
            f.write(article.get("sdxl_image_prompt", ""))
            f.write("\n\n---\n\n")


# Step 9-1. SDXL 이미지 프롬프트를 기반으로 Hugging Face에서 이미지를 생성합니다.
def generate_huggingface_image(article, run_dir):
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(".env 파일에 HF_TOKEN을 먼저 입력하세요.")

    if not article.get("sdxl_image_prompt"):
        article["image_path"] = ""
        article["image_generation_status"] = "skipped_no_sdxl_prompt"
        return article

    output_dir = run_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"article_{article['id']}.png"

    client = InferenceClient(token=hf_token)

    image = client.text_to_image(
        prompt=article["sdxl_image_prompt"],
        negative_prompt=(
            "text, watermark, logo, low quality, blurry, distorted face, "
            "extra fingers, oversaturated, artificial glow"
        ),
        model="stabilityai/stable-diffusion-xl-base-1.0",
        width=1024,
        height=1280,
        num_inference_steps=30,
        guidance_scale=7.5,
    )

    image.save(image_path)

    article["image_path"] = str(image_path)
    article["image_generation_status"] = "success"

    return article


# Step 9-2. 선택된 기사들에 대해 Hugging Face 이미지를 생성합니다.
def generate_huggingface_images(selected_articles, run_dir):
    for article in selected_articles:
        print(f"Hugging Face 이미지 생성 중: {article['title'][:30]}...")
        generate_huggingface_image(article, run_dir)

    return selected_articles

# Step 9-3. 생성된 이미지 하단에 그라데이션과 기사 텍스트를 합성합니다.
def load_korean_font(size, bold=False):
    if bold:
        font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    else:
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

    fallback_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"

    try:
        return ImageFont.truetype(font_path, size=size)
    except OSError:
        return ImageFont.truetype(fallback_path, size=size)
\

def apply_bottom_gradient(image):
    image = image.convert("RGBA")
    width, height = image.size
    gradient = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gradient_pixels = gradient.load()
    start_y = int(height * 0.68)

    for y in range(start_y, height):
        progress = (y - start_y) / max(height - start_y, 1)
        alpha = int(235 * progress)
        for x in range(width):
            gradient_pixels[x, y] = (0, 0, 0, alpha)

    return Image.alpha_composite(image, gradient)


def text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def wrap_text(draw, text, font, max_width, max_lines=2):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

        if len(lines) >= max_lines:
            break

    if current and len(lines) < max_lines:
        lines.append(current)

    if len(lines) == max_lines:
        remaining_text = " ".join(words)
        joined = " ".join(lines)
        if len(joined) < len(remaining_text):
            while lines[-1] and text_width(draw, lines[-1] + "...", font) > max_width:
                lines[-1] = lines[-1][:-1].rstrip()
            lines[-1] = lines[-1] + "..."

    return lines


def clean_article_title(title):
    if " - " in title:
        return title.rsplit(" - ", 1)[0]
    return title

def extract_poster_title(article):
    caption = article.get("instagram_caption", "")

    for line in caption.splitlines():
        line = line.strip()
        if line:
            return line

    return clean_article_title(article.get("title", ""))

def render_news_image_overlay(article):
    image_path = article.get("image_path")
    if not image_path:
        article["final_image_path"] = ""
        article["image_overlay_status"] = "skipped_no_image"
        return article

    input_path = Path(image_path)
    if not input_path.exists():
        article["final_image_path"] = ""
        article["image_overlay_status"] = "image_file_missing"
        return article

    image = Image.open(input_path)
    image = apply_bottom_gradient(image)
    draw = ImageDraw.Draw(image)

    label_font = load_korean_font(45, bold=True)
    title_font = load_korean_font(55, bold=True)
    footer_font = load_korean_font(35)

    x = 75
    label_y = 970
    title_y = 1040
    footer_y = 1190
    max_width = image.size[0] - (x * 2)

    label = "[속보]"
    title = extract_poster_title(article)
    footer = f"출처: {article.get('source', '')} | {datetime.now().strftime('%Y.%m.%d')}"

    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        draw.text((x + dx, label_y + dy), label, fill="#FFFFFF", font=label_font)

    title_lines = wrap_text(draw, title, title_font, max_width=max_width, max_lines=2)
    for idx, line in enumerate(title_lines):
        y = title_y + (idx * 70)

        for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]:
            draw.text((x + dx, y + dy), line, fill="#FFFFFF", font=title_font)

    draw.text((x, footer_y), footer, fill=(221, 221, 221, 215), font=footer_font)

    final_path = input_path.with_name(f"{input_path.stem}_final{input_path.suffix}")
    image.convert("RGB").save(final_path, quality=95)

    article["final_image_path"] = str(final_path)
    article["image_overlay_status"] = "success"
    return article


def render_news_image_overlays(selected_articles):
    for article in selected_articles:
        print(f"이미지 텍스트 합성 중: {article['title'][:30]}...")
        render_news_image_overlay(article)

    return selected_articles

def is_article_complete(article):
    return (
        article.get("status") == "success"
        and article.get("instagram_caption_status") == "success"
        and article.get("sdxl_image_prompt_status") == "success"
        and article.get("image_generation_status") == "success"
        and article.get("image_overlay_status") == "success"
        and bool(article.get("final_image_path"))
    )


def process_content_pipeline(selected_articles, run_dir):
    selected_articles = resolve_selected_article_links(selected_articles)
    selected_articles = fetch_selected_article_bodies(selected_articles)

    selected_articles = generate_instagram_captions(selected_articles)
    save_instagram_captions(selected_articles, run_dir)

    selected_articles = generate_sdxl_image_prompts(selected_articles)
    save_sdxl_image_prompts(selected_articles, run_dir)

    selected_articles = generate_huggingface_images(selected_articles, run_dir)
    selected_articles = render_news_image_overlays(selected_articles)
    save_generated_images(selected_articles, run_dir)

    return selected_articles

def retry_failed_categories_with_backup(selected_articles, run_dir):
    final_articles = []
    failed_categories = []

    for article in selected_articles:
        if is_article_complete(article):
            final_articles.append(article)
            continue

        backup_article = article.get("backup_article")

        if not backup_article:
            failed_categories.append(
                {
                    "category": article.get("category", ""),
                    "primary_id": article.get("id", ""),
                    "backup_id": "",
                    "reason": "primary_failed_no_backup",
                }
            )
            continue

        print(f"1순위 실패, 2순위 기사로 재시도: {article['category']}")

        backup_article["selection_rank"] = "backup"
        backup_article["backup_article"] = None

        processed_backup = process_content_pipeline([backup_article], run_dir)[0]

        if is_article_complete(processed_backup):
            final_articles.append(processed_backup)
        else:
            failed_categories.append(
                {
                    "category": article.get("category", ""),
                    "primary_id": article.get("id", ""),
                    "backup_id": backup_article.get("id", ""),
                    "reason": "primary_and_backup_failed",
                }
            )

    save_failed_categories(failed_categories, run_dir)

    return final_articles

def save_failed_categories(failed_categories, run_dir):
    with open(run_dir / "failed_categories.txt", "w", encoding="utf-8") as f:
        for item in failed_categories:
            f.write(f"Category: {item['category']}\n")
            f.write(f"Primary ID: {item['primary_id']}\n")
            f.write(f"Backup ID: {item['backup_id']}\n")
            f.write(f"Reason: {item['reason']}\n")
            f.write("\n---\n\n")

def save_generated_images(selected_articles, run_dir):
    with open(run_dir / "generated_images.txt", "w", encoding="utf-8") as f:
        for article in selected_articles:
            f.write(f"ID: {article['id']}\n")
            f.write(f"Category: {article['category']}\n")
            f.write(f"Title: {article['title']}\n")
            f.write(f"Status: {article.get('image_generation_status', '')}\n")
            f.write(f"Image Path: {article.get('image_path', '')}\n")
            f.write(f"Final Image Path: {article.get('final_image_path', '')}\n")
            f.write(f"Overlay Status: {article.get('image_overlay_status', '')}\n")
            f.write("\n---\n\n")

# Step 9-1. 선택된 기사 메타데이터를 저장합니다.
def save_selected_news(selected_articles, run_dir):
    with open(run_dir / "selected_news.txt", "w", encoding="utf-8") as f:
        for article in selected_articles:
            f.write(f"ID: {article['id']}\n")
            f.write(f"Category: {article['category']}\n")
            f.write(f"Title: {article['title']}\n")
            f.write(f"Source: {article['source']}\n")
            f.write(f"Google Link: {article['google_link']}\n")
            f.write(f"Resolved Link: {article.get('resolved_link', '')}\n")
            f.write(f"Status: {article.get('status', '')}\n")
            f.write(f"Instagram Caption Status: {article.get('instagram_caption_status', '')}\n")
            f.write("\n---\n\n")


# Step 9-2. 본문, 인스타 캡션, SDXL 이미지 프롬프트까지 저장합니다.
def save_selected_articles(selected_articles, run_dir):
    with open(run_dir / "selected_articles.txt", "w", encoding="utf-8") as f:
        for article in selected_articles:
            f.write(f"ID: {article['id']}\n")
            f.write(f"Category: {article['category']}\n")
            f.write(f"Title: {article['title']}\n")
            f.write(f"Source: {article['source']}\n")
            f.write(f"Google Link: {article['google_link']}\n")
            f.write(f"Resolved Link: {article.get('resolved_link', '')}\n")
            f.write(f"Status: {article.get('status', '')}\n")
            f.write(f"Instagram Caption Status: {article.get('instagram_caption_status', '')}\n")
            f.write("Body:\n")
            f.write(article.get("body", ""))
            f.write("\n\nInstagram Caption:\n")
            f.write(article.get("instagram_caption", ""))
            f.write("\n\nSDXL Image Prompt:\n")
            f.write(article.get("sdxl_image_prompt", ""))
            f.write("\n\nGenerated Image Path:\n")
            f.write(article.get("image_path", ""))
            f.write("\n\nFinal Image Path:\n")
            f.write(article.get("final_image_path", ""))
            f.write("\n\n---\n\n")
            
# Step 10. 업로드 또는 최종 완료된 기사 기록을 history.jsonl에 누적 저장합니다.
def append_publish_history(selected_articles, status="ready"):
    published_at = datetime.now().isoformat(timespec="seconds")

    with open("history.jsonl", "a", encoding="utf-8") as f:
        for article in selected_articles:
            record = {
                "published_at": published_at,
                "status": status,
                "category": article.get("category", ""),
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "google_link": article.get("google_link", ""),
                "resolved_link": article.get("resolved_link", ""),
                "instagram_post_id": article.get("instagram_post_id", ""),
                "final_image_path": article.get("final_image_path", ""),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
# outputs 폴더에서 최근 keep_days일보다 오래된 날짜 폴더를 삭제합니다.
def cleanup_old_outputs(keep_days=3):
    outputs_dir = Path("outputs")

    if not outputs_dir.exists():
        return

    today = datetime.now().date()

    for run_dir in outputs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        try:
            run_date = datetime.strptime(run_dir.name, "%Y-%m-%d").date()
        except ValueError:
            continue

        age_days = (today - run_date).days

        if age_days >= keep_days:
            shutil.rmtree(run_dir)
            print(f"오래된 outputs 폴더 삭제: {run_dir}")
            
# 인스타 업로드 성공 후 실행할 후처리 함수입니다.
def handle_publish_success(published_articles):
    append_publish_history(published_articles, status="published")
    cleanup_old_outputs(keep_days=3)

# Step 10. 전체 파이프라인을 실행합니다.
if __name__ == "__main__":
    run_dir = create_run_dir()
    news_list = fetch_top_news()

    for news in news_list[:3]:
        print(news)

    print(f"\n필터링 완료. 총 {len(news_list)}개의 신선한 뉴스를 확보했습니다.\n")

    if len(news_list) > 0:
        print("--- [Gemini 선정 결과] ---")
        selected_result = select_best_articles(news_list)
        print(selected_result)

        with open(run_dir / "gemini_selected_result.txt", "w", encoding="utf-8") as f:
            f.write(selected_result)

        selected_articles = match_selected_articles(selected_result, news_list)

        selected_articles = process_content_pipeline(selected_articles, run_dir)
        selected_articles = retry_failed_categories_with_backup(selected_articles, run_dir)

        save_selected_news(selected_articles, run_dir)
        save_selected_articles(selected_articles, run_dir)

        print("\n[완료] 오늘 콘텐츠 생성 파이프라인이 끝났습니다.")
        print(f"산출물 저장 위치: {run_dir}")

    else:
        print("수집된 뉴스가 없습니다. 구글 뉴스 연결 상태를 확인하세요.")
