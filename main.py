import os
from pygooglenews import GoogleNews


def fetch_top_news():
    print("[Step 1] 글로벌 구글 뉴스 데이터 수집 (단일 바구니 필터링 적용)...")

    # 과거 이력 및 중복 확인
    seen_links = set()

    # 과거 이력
    if os.path.exists("history.txt"):
        with open("history.txt", "r", encoding="utf-8") as f:
            seen_links.update(line.strip() for line in f.readlines())
        print(f"어제 기록된 뉴스 {len(seen_links)}건을 블랙리스트에 선탑재했습니다.")
    else:
        print("어제 기록이 없습니다.")

    gn_kr = GoogleNews(lang="ko", country="KR")
    gn_us = GoogleNews(lang="en", country="US")

    raw_news = []

    def add_news(entries, category_name):
        added_count = 0
        for entry in entries:
            if entry.link in seen_links:
                continue

            raw_news.append(
                {"title": entry.title, "link": entry.link, "category": category_name}
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


# --- 테스트 실행 블록 ---
if __name__ == "__main__":
    # 1단계: 뉴스 수집 및 이중 필터링 실행
    news_list = fetch_top_news()

    # 데이터 결과 보고
    print(f"\n필터링 완료. 총 {len(news_list)}개의 신선한 뉴스를 확보했습니다.\n")

    if len(news_list) > 0:
        # 카테고리별 샘플 출력 (검증용)
        print("--- [카테고리별 샘플 확인] ---")
        categories = ["종합(KR)", "경제(KR)", "경제(US)"]
        for cat in categories:
            cat_news = [n for n in news_list if n["category"] == cat][:1]
            for n in cat_news:
                print(f"[{n['category']}] {n['title']}")

        # history.txt 생성
        with open("history.txt", "w", encoding="utf-8") as f:
            f.write(news_list[0]["link"] + "\n")

        print(
            f"\n[테스트 알림] '{news_list[0]['title'][:20]}...' 기사를 history.txt에 기록했습니다."
        )
        print("한 번 더 실행하면 위 기사가 블랙리스트에 의해 걸러지는지 확인하세요!")
    else:
        print("수집된 뉴스가 없습니다. 구글 뉴스 연결 상태를 확인하세요.")
