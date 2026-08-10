import os
import json
import time
import requests
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DATA_DIR = "data"
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")

def get_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return [m["market"] for m in res.json() if m["market"].startswith("KRW-")]
    except Exception as e:
        print(f"⚠️ 마켓 목록 조회 실패: {e}")
    return []

def fetch_5m_candles_deep(market, target_count=2000):
    all_candles = []
    to_param = ""
    retry_count = 0

    while len(all_candles) < target_count:
        req_count = min(200, target_count - len(all_candles))
        url = f"https://api.upbit.com/v1/candles/minutes/5?market={market}&count={req_count}"
        if to_param:
            url += f"&to={to_param}"

        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 429:
                time.sleep(0.5)
                continue
            elif res.status_code != 200:
                retry_count += 1
                if retry_count > 3: break
                time.sleep(0.2)
                continue

            data = res.json()
            if not data: break

            all_candles.extend(data)
            if len(data) < req_count: break

            to_param = data[-1]["candle_date_time_utc"]
            retry_count = 0
            time.sleep(0.08)
        except Exception:
            time.sleep(0.2)
            continue

    return all_candles

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    markets = get_krw_markets()

    all_price_patterns = []
    all_volume_patterns = []
    print(f"🔄 [자율 진화 1단계] 최신 {len(markets)}개 종목 대상 5분 봉 대량 학습 및 패턴 재생성...")

    for idx, market in enumerate(markets):
        candles = fetch_5m_candles_deep(market, target_count=2000)
        if len(candles) < 100: continue

        df = pd.DataFrame(candles).sort_values("timestamp").reset_index(drop=True)

        for i in range(24, len(df) - 18):
            base_price = df.iloc[i]["trade_price"]
           
            if df.iloc[i]["candle_acc_trade_price"] < 50_000_000:
                continue

            if base_price > 0:
                future_max_price = df.iloc[i + 1 : i + 7]["trade_price"].max()
                future_min_price = df.iloc[i + 1 : i + 19]["trade_price"].min()

                surge_rate = (future_max_price - base_price) / base_price
                post_drop_rate = (future_min_price - base_price) / base_price

                if surge_rate >= 0.08 and post_drop_rate >= -0.02:
                    pre_prices = df.iloc[i - 24 : i]["trade_price"].values
                    pre_volumes = df.iloc[i - 24 : i]["candle_acc_trade_volume"].values

                    log_volumes = np.log1p(pre_volumes)

                    p_min, p_max = pre_prices.min(), pre_prices.max()
                    v_min, v_max = log_volumes.min(), log_volumes.max()

                    if p_max > p_min and v_max > v_min:
                        norm_prices = (pre_prices - p_min) / (p_max - p_min + 1e-8)
                        norm_volumes = (log_volumes - v_min) / (v_max - v_min + 1e-8)

                        all_price_patterns.append(norm_prices)
                        all_volume_patterns.append(norm_volumes)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(markets):
            print(f"⌛ 패턴 진행률: {idx + 1}/{len(markets)} 완료... (수집 패턴: {len(all_price_patterns)}개)")

    if len(all_price_patterns) >= 3:
        kmeans_p = KMeans(n_clusters=3, random_state=42, n_init=10).fit(all_price_patterns)
        kmeans_v = KMeans(n_clusters=3, random_state=42, n_init=10).fit(all_volume_patterns)

        pattern_data = {
            "golden_patterns": kmeans_p.cluster_centers_.tolist(),
            "golden_volume_patterns": kmeans_v.cluster_centers_.tolist(),
            "sample_count": len(all_price_patterns),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(PATTERN_FILE, "w", encoding="utf-8") as f:
            json.dump(pattern_data, f, ensure_ascii=False, indent=4)

        print(f"✅ 패턴 자동 갱신 완료! ('{PATTERN_FILE}')")

if __name__ == "__main__":
    main()
