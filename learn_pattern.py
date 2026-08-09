import json
import os
import time
import numpy as np
import pandas as pd
import requests

DATA_DIR = "data"
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")


def get_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            markets = res.json()
            return [
                m["market"] for m in markets if m["market"].startswith("KRW-")
            ]
    except Exception as e:
        print(f"⚠️ 마켓 목록 조회 실패: {e}")
    return []


def fetch_5m_candles(market, count=200):
    """5분 봉 연속 데이터 수집"""
    url = f"https://api.upbit.com/v1/candles/minutes/5?market={market}&count={count}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    markets = get_markets()

    all_surge_patterns = []
    print(
        f"🔍 총 {len(markets)}개 종목 대상 5분 봉 기준 '폭등 직전 패턴' 학습 시작..."
    )

    for idx, market in enumerate(markets):
        # API Rate Limit 방지를 위한 지연 (초당 10회 안정성 유지)
        time.sleep(0.1)

        candles = fetch_5m_candles(market, count=200)
        if len(candles) < 50:
            continue

        # 과거 순으로 정렬 (Timestamp 오름차순)
        df = pd.DataFrame(candles).sort_values("timestamp").reset_index(drop=True)

        # 5분 봉 연속 구간 중, 향후 30분(6개 캔들) 내 +10%~20% 이상 급등한 지점 탐색
        # (5분 봉 단타에서는 1시간 내 10~20% 상승도 강력한 단기 폭등 시그널임)
        for i in range(12, len(df) - 6):
            base_price = df.iloc[i]["trade_price"]
            future_max_price = df.iloc[i + 1 : i + 7]["trade_price"].max()

            if base_price > 0:
                surge_rate = (future_max_price - base_price) / base_price

                # 폭등 직전 12개 캔들(1시간) 패턴 추출 (급등 발생 전 시점)
                if surge_rate >= 0.10: # 5분 봉 단타 특성에 맞게 10% 이상 급등 탐지
                    pre_prices = df.iloc[i - 12 : i]["trade_price"].values
                    p_min, p_max = pre_prices.min(), pre_prices.max()

                    if p_max > p_min:
                        # 0 ~ 1 정규화 (realtime.py와 동일 프레임)
                        norm_prices = (pre_prices - p_min) / (p_max - p_min)
                        all_surge_patterns.append(norm_prices)

        if (idx + 1) % 20 == 0:
            print(f"⌛ 진행률: {idx + 1}/{len(markets)} 코인 스캔 완료...")

    if all_surge_patterns:
        # 추출된 폭등 전조 패턴들의 평균 계산
        golden_pattern = np.mean(all_surge_patterns, axis=0).tolist()

        pattern_data = {
            "golden_pattern": golden_pattern,
            "sample_count": len(all_surge_patterns),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(PATTERN_FILE, "w", encoding="utf-8") as f:
            json.dump(pattern_data, f, ensure_ascii=False, indent=4)

        print(
            f"✅ 총 {len(all_surge_patterns)}개의 폭등 전조 샘플 학습 완료! 'golden_pattern.json' 최신화 완료."
        )
    else:
        print(
            "⚠️ 조건에 부합하는 급등 패턴을 찾지 못해 기존 패턴 파일을 유지합니다."
        )


if __name__ == "__main__":
    main()
