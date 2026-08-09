from datetime import datetime, timezone, timedelta
import json
import os

HISTORY_FILE = os.path.join("data", "history_db.json")
KST = timezone(timedelta(hours=9))

# 조회할 대상 코인 티커 목록 (KRW 기준)
TARGET_TICKERS = ["KRW-LA", "KRW-ESP", "KRW-PUMP", "KRW-CAP", "KRW-MOMENTUM"]


def check_recommendation_history():
    if not os.path.exists(HISTORY_FILE):
        print(f"❌ 히스토리 파일이 존재하지 않습니다: {HISTORY_FILE}")
        return

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history_data = json.load(f)

    print("=== 📊 주요 급등 종목 추천 및 TOP 10 진입 이력 조회 ===\n")

    for ticker in TARGET_TICKERS:
        records = history_data.get(ticker, [])
        if not records:
            # 티커 형식이 다를 경우(예: 'LA', 'ESP') 대비
            short_ticker = ticker.replace("KRW-", "")
            records = history_data.get(short_ticker, [])

        if not records:
            print(f"🔍 [{ticker}] 기록 없음")
            continue

        print(f"📌 [{ticker}] 포착 이력 (총 {len(records)}건):")

        # TOP 10에 들어간 시점 위주로 출력
        top_ranks = [
            r for r in records if r.get("rank", 99) <= 10
        ]

        if not top_ranks:
            print("   - TOP 10 랭킹 진입 기록 없음 (하위권 유지)")
        else:
            for item in top_ranks:
                dt = datetime.fromtimestamp(item["timestamp"], tz=KST)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"   • {time_str} KST | 랭킹: {item.get('rank')}위 | 점수: {item.get('score')}점 | 포착가: {item.get('price'):,} KRW"
                )
        print("-" * 50)


if __name__ == "__main__":
    check_recommendation_history()
