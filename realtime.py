import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import threading
import time
import websockets
from fastdtw import fastdtw
import numpy as np
import pandas as pd
import requests
from scipy.spatial.distance import euclidean

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
DOCS_DIR = "docs"
HTML_OUTPUT = os.path.join(DOCS_DIR, "index.html")

KST = timezone(timedelta(hours=9))


# ==========================================
# [1순위 반영] WebSocket 실시간 데이터 수신 매니저
# ==========================================
class UpbitWebSocketManager:

    def __init__(self, markets):
        self.markets = markets
        self.ticker_data = {}
        self.is_running = False

    async def _connect_websocket(self):
        url = "wss://api.upbit.com/websocket/v1"
        subscribe_fmt = [
            {"ticket": "TEST_TICKET"},
            {"type": "ticker", "codes": self.markets},
        ]

        while self.is_running:
            try:
                async with websockets.connect(url) as websocket:
                    await websocket.send(json.dumps(subscribe_fmt))
                    while self.is_running:
                        data = await websocket.recv()
                        res = json.loads(data)
                        code = res.get("code")
                        if code:
                            self.ticker_data[code] = {
                                "trade_price": res.get("trade_price"),
                                "signed_change_rate": res.get(
                                    "signed_change_rate", 0
                                )
                                * 100,
                                "acc_trade_price_24h": res.get(
                                    "acc_trade_price_24h", 0
                                ),
                                "high_price": res.get("high_price"),
                                "low_price": res.get("low_price"),
                            }
            except Exception as e:
                await asyncio.sleep(2)

    def _start_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect_websocket())

    def start(self):
        self.is_running = True
        t = threading.Thread(target=self._start_async_loop, daemon=True)
        t.start()

    def stop(self):
        self.is_running = False


# ==========================================
# [3순위 반영] DTW 기반 패턴 유사도 산출 함수
# ==========================================
def calculate_dtw_similarity(seq1, seq2):
    """DTW(Dynamic Time Warping) 기반 패턴 유사도 산출 (1차원 평탄화 보장)"""
    try:
        # 입력 데이터가 리스트나 2차원 형태일 경우 1차원으로 강제 변환
        s1 = np.asarray(seq1, dtype=np.float64).ravel()
        s2 = np.asarray(seq2, dtype=np.float64).ravel()

        if len(s1) == 0 or len(s2) == 0:
            return 0.0

        distance, _ = fastdtw(s1, s2, dist=euclidean)
        # 24개 데이터 정규화 규격 상 최대 거리는 약 5.0 근방
        similarity = max(0.0, (1.0 - (distance / 5.0))) * 100
        return round(similarity, 1)
    except Exception as e:
        print(f"⚠️ DTW 계산 중 오류: {e}")
        return 0.0


def fetch_ai_recommendations():
    url = "https://raw.githubusercontent.com/mdog002-wq/upbit-a/main/docs/ai_recommend_tracker.json"
    refined_set = set()
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            raw_tickers = []
            if isinstance(data, dict):
                raw_tickers = data.get("recommended_tickers", []) or data.get(
                    "recommended_coins", []
                )
            elif isinstance(data, list):
                raw_tickers = data

            for t in raw_tickers:
                if t and isinstance(t, str):
                    t_str = t.strip().upper()
                    refined_set.add(t_str)
                    refined_set.add(f"KRW-{t_str.replace('KRW-', '')}")
            return refined_set
    except Exception:
        pass
    return set()


def fetch_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            markets = res.json()
            krw_markets = [
                m["market"] for m in markets if m["market"].startswith("KRW-")
            ]
            market_names = {
                m["market"]: m["korean_name"]
                for m in markets
                if m["market"].startswith("KRW-")
            }
            return krw_markets, market_names
    except Exception:
        pass
    return [], {}


def fetch_5m_candles(market, count=120):
    url = f"https://api.upbit.com/v1/candles/minutes/5?market={market}&count={count}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50.0


def check_btc_status():
    try:
        btc_candles = fetch_5m_candles("KRW-BTC", count=24)
        if len(btc_candles) < 24:
            return "NEUTRAL (보통)", 1.0
        df = pd.DataFrame(btc_candles).sort_values("timestamp")
        btc_change = (
            (df.iloc[-1]["trade_price"] - df.iloc[0]["opening_price"])
            / df.iloc[0]["opening_price"]
        ) * 100
        if btc_change <= -2.0:
            return "BEAR (하락장 경고)", 0.85
        elif btc_change >= 1.5:
            return "BULL (강세장)", 1.05
        return "NEUTRAL (보통)", 1.0
    except Exception:
        return "NEUTRAL (보통)", 1.0


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


def analyze_single_coin(
    market,
    k_name,
    ideal_price_pattern,
    ideal_vol_pattern,
    history_db,
    weights,
    btc_multiplier,
    ws_data,
    is_ai_recommended,
):
    ticker = market.replace("KRW-", "")
    candles = fetch_5m_candles(market, count=120)
    if len(candles) < 60:
        return None

    df = pd.DataFrame(candles).sort_values("timestamp").reset_index(drop=True)

    # 웹소켓 체결 시세 적용
    if ws_data and "trade_price" in ws_data:
        current_price = ws_data["trade_price"]
        change_rate = ws_data["signed_change_rate"]
    else:
        current_price = df.iloc[-1]["trade_price"]
        change_rate = (
            (current_price - df.iloc[-1]["prev_closing_price"])
            / df.iloc[-1]["prev_closing_price"]
        ) * 100

    df_2h = df.iloc[-24:].copy().reset_index(drop=True)

    # ==========================================
    # [2&3순위] DTW 활용 가격(70%) + 거래량(30%) 합성 유사도
    # ==========================================
    prices = df_2h["trade_price"].values
    volumes = df_2h["candle_acc_trade_volume"].values

    norm_prices = (prices - prices.min()) / (
        prices.max() - prices.min() + 1e-8
    )
    norm_volumes = (volumes - volumes.min()) / (
        volumes.max() - volumes.min() + 1e-8
    )

    price_sim = calculate_dtw_similarity(norm_prices, ideal_price_pattern)
    vol_sim = calculate_dtw_similarity(norm_volumes, ideal_vol_pattern)

    combined_pattern_sim = round(price_sim * 0.7 + vol_sim * 0.3, 1)

    # 보조 지표 계산
    accumulation_score = min(
        100.0, (df_2h["candle_acc_trade_volume"].std() / (df_2h["candle_acc_trade_volume"].mean() + 1e-8)) * 40
    )
    rsi = calculate_rsi(df["trade_price"])

    high_24h = df["high_price"].max()
    breakout_score = (
        100.0 if current_price >= high_24h else (current_price / high_24h) * 100
    )

    recent_vol = df.iloc[-3:]["candle_acc_trade_volume"].sum()
    avg_vol = df.iloc[-36:-3]["candle_acc_trade_volume"].sum() / 33 * 3
    vol_surge_score = min(100.0, (recent_vol / (avg_vol + 1e-8)) * 25.0)

    acc_24h_krw = df["candle_acc_trade_price"].sum()
    liquidity_index = min(
        100.0, max(0.0, (np.log10(acc_24h_krw + 1e-8) - 7) * 20)
    )

    base_score = (
        combined_pattern_sim * weights.get("w_pattern", 0.15)
        + accumulation_score * weights.get("w_accumulation", 0.15)
        + breakout_score * weights.get("w_breakout", 0.20)
        + vol_surge_score * weights.get("w_vol_surge", 0.25)
        + min(100.0, change_rate * 3.33) * weights.get("w_daily_momentum", 0.25)
    )

    final_score = base_score * btc_multiplier
    if is_ai_recommended:
        final_score *= 1.05

    return {
        "market": market,
        "ticker": ticker,
        "name": k_name,
        "current_price": current_price,
        "change_rate": round(change_rate, 2),
        "pattern_similarity": combined_pattern_sim,
        "accumulation_score": round(accumulation_score, 1),
        "score": round(final_score, 2),
        "rsi": round(rsi, 1),
        "liquidity_index": round(liquidity_index, 1),
        "is_ai_recommended": is_ai_recommended,
    }


def generate_dashboard_html(
    analysis_results, current_time_str, btc_status, html_path
):
    rows_list = []
    for item in analysis_results:
        change_class = (
            "plus"
            if item["change_rate"] > 0
            else ("minus" if item["change_rate"] < 0 else "")
        )
        change_sign = "+" if item["change_rate"] > 0 else ""
        ai_badge = (
            '<span style="background:#e03131;color:#fff;font-size:10px;padding:2px 4px;border-radius:3px;margin-left:4px;">AI</span>'
            if item["is_ai_recommended"]
            else ""
        )

        row = f"""
<tr>
<td><b>{item['rank']}</b></td>
<td><b>{item['name']}</b> ({item['ticker']}){ai_badge}</td>
<td>{item['current_price']:,}</td>
<td class="{change_class}">{change_sign}{item['change_rate']}%</td>
<td>{item['rsi']}</td>
<td><b>{item['pattern_similarity']}%</b></td>
<td>{item['accumulation_score']}점</td>
<td><b>{item['score']}점</b></td>
</tr>"""
        rows_list.append(row)

    rows_html = "".join(rows_list)

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>실시간 고도화 급등주 대시보드</title>
<meta http-equiv="refresh" content="300">
<style>
body {{ font-family: sans-serif; background: #f8f9fa; padding: 20px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; }}
th, td {{ padding: 10px; text-align: center; border-bottom: 1px solid #ddd; }}
th {{ background: #e9ecef; }}
.plus {{ color: #e03131; font-weight: bold; }}
.minus {{ color: #1971c2; font-weight: bold; }}
</style>
</head>
<body>
<h2>🚀 실시간 DTW + 웹소켓 고도화 대시보드</h2>
<p>업데이트: <b>{current_time_str}</b> | 비트코인: <b>{btc_status}</b></p>
<table>
<thead>
<tr><th>순위</th><th>코인명</th><th>현재가(KRW)</th><th>등락률</th><th>RSI</th><th>DTW패턴유사도</th><th>세력매집</th><th>최종점수</th></tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>"""

    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    krw_markets, market_names = fetch_krw_markets()

    # 1. 웹소켓 매니저 가동 (배경에서 실시간 수신)
    ws_manager = UpbitWebSocketManager(krw_markets)
    ws_manager.start()
    print("🌐 웹소켓 연결 시작... (시세 데이터 실시간 가공 중)")
    time.sleep(3) # 초기 웹소켓 수신 대기

    ai_recommend_set = fetch_ai_recommendations()
    btc_status, btc_multiplier = check_btc_status()
    weights = load_json(
        WEIGHTS_FILE,
        {
            "w_pattern": 0.15,
            "w_accumulation": 0.15,
            "w_breakout": 0.20,
            "w_vol_surge": 0.25,
            "w_daily_momentum": 0.25,
        },
    )

     # 기존 pattern_data 로드 부분을 아래와 같이 1차원 평탄화(flatten)되도록 수정
    pattern_data = load_json(PATTERN_FILE, {})

    ideal_price_pattern = np.array(
        pattern_data.get("golden_pattern", np.linspace(0.2, 1.0, 24)),
        dtype=np.float64,
    ).ravel()

    ideal_vol_pattern = np.array(
        pattern_data.get("golden_volume_pattern", np.linspace(0.1, 1.0, 24)),
        dtype=np.float64,
    ).ravel()

    results = []
    print("🔍 1, 2, 3순위 고도화 로직 기반 종목 분석 시작...")

    for market in krw_markets:
        ws_data = ws_manager.ticker_data.get(market, {})
        res = analyze_single_coin(
            market,
            market_names.get(market, market),
            ideal_price_pattern,
            ideal_vol_pattern,
            {},
            weights,
            btc_multiplier,
            ws_data,
            (market in ai_recommend_set or market.replace("KRW-", "") in ai_recommend_set),
        )
        if res:
            results.append(res)

    ws_manager.stop()

    results.sort(key=lambda x: x["score"], reverse=True)
    for idx, item in enumerate(results):
        item["rank"] = idx + 1

    current_time_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    generate_dashboard_html(
        results, current_time_str, btc_status, HTML_OUTPUT
    )
    print("🎨 고도화 대시보드 HTML 생성 완료!")


if __name__ == "__main__":
    main()
