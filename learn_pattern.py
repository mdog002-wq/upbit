import os
import json
import time
import requests
import numpy as np
import pandas as pd

DATA_DIR = "data"
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")

def get_markets():
    url = "https://api.upbit.com/v1/market/all"
    res = requests.get(url)
    if res.status_code != 200:
        return []
    markets = res.json()
    return [m['market'] for m in markets if m['market'].startswith("KRW-")]

def fetch_daily_candles(market, count=180):
    url = f"https://api.upbit.com/v1/candles/days?market={market}&count={count}"
    res = requests.get(url)
    return res.json() if res.status_code == 200 else []

def fetch_15m_candles_before(market, to_date):
    url = f"https://api.upbit.com/v1/candles/minutes/15?market={market}&to={to_date}&count=24"
    res = requests.get(url)
    return res.json() if res.status_code == 200 else []

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    markets = get_markets()
    
    all_surge_patterns = []
    print(f"🔍 총 {len(markets)}개 종목에 대해 과거 6개월간 20% 이상 급등 패턴 탐색 시작...")

    for market in markets:
        daily_candles = fetch_daily_candles(market, count=180)
        time.sleep(0.05)
        
        for candle in daily_candles:
            open_p = candle.get('opening_price', 0)
            high_p = candle.get('high_price', 0)
            
            if open_p > 0 and ((high_p - open_p) / open_p) >= 0.20:
                target_date = candle['candle_date_time_utc'] + "Z"
                m15_candles = fetch_15m_candles_before(market, target_date)
                time.sleep(0.05)
                
                if len(m15_candles) == 24:
                    df = pd.DataFrame(m15_candles).sort_values('timestamp')
                    prices = df['trade_price'].values
                    
                    price_min = prices.min()
                    price_max = prices.max()
                    if price_max != price_min:
                        norm_prices = (prices - price_min) / (price_max - price_min)
                        all_surge_patterns.append(norm_prices)

    if all_surge_patterns:
        golden_pattern = np.mean(all_surge_patterns, axis=0).tolist()
        
        with open(PATTERN_FILE, 'w', encoding='utf-8') as f:
            json.dump({"golden_pattern": golden_pattern, "sample_count": len(all_surge_patterns)}, f, ensure_ascii=False, indent=4)
        
        print(f"✅ 총 {len(all_surge_patterns)}개의 급등 전조 패턴 분석 완료! 'golden_pattern.json' 저장 완료.")
    else:
        print("⚠️ 조건에 부합하는 급등 패턴을 찾지 못해 기존 데이터를 유지합니다.")

if __name__ == "__main__":
    main()
