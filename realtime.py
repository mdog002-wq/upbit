import asyncio
import json
import websockets
import pandas as pd
import numpy as np
import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastdtw import fastdtw

# ============================================================
# 경로 설정 (docs/ 폴더 경로 지정으로 웹 대시보드 완벽 연동)
# ============================================================
DATA_DIR = "data"
DOCS_DIR = "docs"

# 대시보드 웹사이트가 읽는 핵심 파일 경로들
HISTORY_FILE = os.path.join(DOCS_DIR, "history_db.json")
DASHBOARD_FILE = os.path.join(DOCS_DIR, "dashboard_data.json")

WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
REMOTE_TRACKER_URL = "https://raw.githubusercontent.com/mdog002-wq/upbit/main/docs/ai_recommend_tracker.json"

# 실시간 데이터를 관리할 인메모리 딕셔너리
REALTIME_CACHE = {}


def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def calculate_dtw_similarity(seq1, seq2):
    try:
        s1 = np.asarray(seq1, dtype=np.float64).reshape(-1)
        s2 = np.asarray(seq2, dtype=np.float64).reshape(-1)
        min_len = min(len(s1), len(s2))
        if min_len == 0:
            return 0.0
        s1, s2 = s1[-min_len:], s2[-min_len:]
        distance, _ = fastdtw(s1, s2, dist=lambda x, y: abs(x - y))
        avg_dist = distance / min_len
        return round(float(np.exp(-1.5 * avg_dist) * 100.0), 1)
    except Exception:
        return 0.0


def calculate_max_dtw(seq1, golden_patterns):
    if not golden_patterns:
        return 0.0
    max_sim = 0.0
    for pattern in golden_patterns:
        sim = calculate_dtw_similarity(seq1, pattern)
        if sim > max_sim:
            max_sim = sim
    return max_sim


def fetch_5m_candles(market, count=120):
    url = f"https://api.upbit.com/v1/candles/minutes/5?market={market}&count={count}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def calculate_atr(df, period=14):
    try:
        high, low, close = df['high_price'], df['low_price'], df['trade_price'].shift(1)
        tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return atr if not np.isnan(atr) else (df['trade_price'].iloc[-1] * 0.015)
    except Exception:
        return df['trade_price'].iloc[-1] * 0.015


def fetch_remote_recommendations():
    try:
        res = requests.get(f"{REMOTE_TRACKER_URL}?t={int(time.time())}", timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data and isinstance(data, list):
                latest = data[-1]
                return [c.get("symbol") for c in latest.get("recommended_coins", []) if c.get("symbol")]
    except Exception:
        pass
    return []


def analyze_single_coin(market, k_name, golden_price_patterns, golden_vol_patterns, weights, recommended_symbols):
    ticker = market.replace("KRW-", "")
    candles = fetch_5m_candles(market, count=120)
    if len(candles) < 60:
        return None

    df = pd.DataFrame(candles).sort_values("timestamp").reset_index(drop=True)
    current_price = df.iloc[-1]["trade_price"]
    change_rate = ((current_price - df.iloc[-1]["prev_closing_price"]) / df.iloc[-1]["prev_closing_price"]) * 100

    df_2h = df.iloc[-24:].copy().reset_index(drop=True)
    prices, volumes = df_2h["trade_price"].values, df_2h["candle_acc_trade_volume"].values

    p_range = (prices.max() - prices.min()) or 1.0
    v_range = (volumes.max() - volumes.min()) or 1.0

    norm_prices = (prices - prices.min()) / p_range
    norm_volumes = (volumes - volumes.min()) / v_range

    price_sim = calculate_max_dtw(norm_prices, golden_price_patterns)
    vol_sim = calculate_max_dtw(norm_volumes, golden_vol_patterns)
    combined_pattern_sim = round(price_sim * 0.7 + vol_sim * 0.3, 1)

    recent_vol = df.iloc[-1]["candle_acc_trade_volume"]
    avg_prev_vol = df.iloc[-21:-1]["candle_acc_trade_volume"].mean()
    vol_cliff_score = min(100.0, max(0.0, (1.0 - (recent_vol / (avg_prev_vol + 1e-8))) * 100.0)) if avg_prev_vol > 0 else 0.0

    df["ma5"] = df["trade_price"].rolling(5).mean()
    df["ma20"] = df["trade_price"].rolling(20).mean()
    df["ma60"] = df["trade_price"].rolling(60).mean()
    last = df.iloc[-1]
    ma_score = 100.0 if last["ma5"] > last["ma20"] > last["ma60"] else (60.0 if last["ma5"] > last["ma20"] else 20.0)

    # RSI 14 계산
    delta = df["trade_price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi_val = float(100 - (100 / (1 + (gain / (loss + 1e-8)).iloc[-1])))

    # 🧬 동적 가중치 기반 종합점수 산출
    base_score = (
        combined_pattern_sim * weights.get("w_pattern", 0.20) +
        vol_cliff_score * weights.get("w_vol_cliff", 0.25) +
        ma_score * weights.get("w_ma_alignment", 0.25) +
        min(100.0, max(0.0, change_rate * 3.33)) * weights.get("w_daily_momentum", 0.10) +
        (current_price / df["high_price"].max() * 100) * weights.get("w_breakout", 0.05)
    )

    if ticker in recommended_symbols:
        base_score += 15.0

    atr = calculate_atr(df)
    tp1 = current_price + (atr * 2.0)
    sl = current_price - (atr * 1.5)

    sc = round(min(100.0, base_score), 2)

    # 🌐 웹 호환을 위한 영문/한글 필드 통합 지정
    return {
        "market": market,
        "ticker": ticker,
        "name": k_name,
        "current_price": current_price,
        "change_rate": round(change_rate, 2),
        "score": sc,
        "rsi": round(rsi_val, 1),
        "pattern_similarity": combined_pattern_sim,
        "tp1": round(tp1, 2),
        "sl": round(sl, 2),
        "is_repo1_recommended": ticker in recommended_symbols,
        # 한글 키 호환용
        "코인명": k_name,
        "심볼": ticker,
        "현재가(KRW)": current_price,
        "종합예측점수": sc,
        "RSI": round(rsi_val, 1),
        "골든패턴유사도(%)": combined_pattern_sim
    }


def update_and_save_dashboard_data():
    """웹 대시보드용 JSON 파일(docs/history_db.json 및 dashboard_data.json) 동시 업데이트"""
    data_list = list(REALTIME_CACHE.values())
    data_list.sort(key=lambda x: x["score"], reverse=True)

    # 웹 연동 파일 양쪽에 동시 저장
    save_json(HISTORY_FILE, data_list)
    save_json(DASHBOARD_FILE, data_list)


async def connect_upbit_websocket(markets):
    url = "wss://api.upbit.com/websocket/v1"
    subscribe_data = [{"ticket": "QUANT_BOT"}, {"type": "ticker", "codes": markets}]
    
    last_save_time = time.time()

    while True:
        try:
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps(subscribe_data))
                print("📡 Upbit 실시간 웹소켓 연결 성공 및 실시간 트래킹 시작!")

                while True:
                    data = await ws.recv()
                    # 바이너리 바이트 데이터를 텍스트로 변환
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')

                    raw = json.loads(data)
                    code = raw.get("code")

                    # 실시간 현재가 및 변동률 메모리 캐시 반영
                    if code in REALTIME_CACHE:
                        c_price = raw.get("trade_price", REALTIME_CACHE[code]["current_price"])
                        signed_change_rate = raw.get("signed_change_rate", 0) * 100

                        REALTIME_CACHE[code]["current_price"] = c_price
                        REALTIME_CACHE[code]["현재가(KRW)"] = c_price
                        REALTIME_CACHE[code]["change_rate"] = round(signed_change_rate, 2)

                        # 3초마다 대시보드 파일로 실시간 반영하여 저장
                        now = time.time()
                        if now - last_save_time >= 3.0:
                            update_and_save_dashboard_data()
                            last_save_time = now

        except Exception as e:
            print(f"⚠️ 웹소켓 연결 재시도 중... ({e})")
            await asyncio.sleep(3)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    weights = load_json(WEIGHTS_FILE, {
        "w_pattern": 0.20, "w_vol_cliff": 0.25, "w_ma_alignment": 0.25,
        "w_vol_surge": 0.15, "w_daily_momentum": 0.10, "w_breakout": 0.05
    })
    print(f"📊 현재 적용된 자율 학습 가중치: {weights}")

    pattern_data = load_json(PATTERN_FILE, {})
    golden_price_patterns = pattern_data.get("golden_patterns", [])
    golden_vol_patterns = pattern_data.get("golden_volume_patterns", [])

    recommended_symbols = fetch_remote_recommendations()

    res = requests.get("https://api.upbit.com/v1/market/all")
    all_krw = [m for m in res.json() if m["market"].startswith("KRW-")]

    analyzed_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                analyze_single_coin, item["market"], item["korean_name"],
                golden_price_patterns, golden_vol_patterns, weights, recommended_symbols
            ) for item in all_krw
        ]
        for f in as_completed(futures):
            r = f.result()
            if r:
                analyzed_results.append(r)

    analyzed_results.sort(key=lambda x: x["score"], reverse=True)

    # 분석 완료 데이터 메모리 캐시에 등록
    for item in analyzed_results:
        REALTIME_CACHE[item["market"]] = item

    # 초기 대시보드 JSON 파일 바로 업데이트
    update_and_save_dashboard_data()
    print(f"✅ 2차 진화형 실시간 스코어링 및 docs/ 파일 저장 완료 (1위: {analyzed_results[0]['ticker']} - {analyzed_results[0]['score']}점)")

    top_markets = [x["market"] for x in analyzed_results[:20]]
    asyncio.run(connect_upbit_websocket(top_markets))


if __name__ == "__main__":
    main()
