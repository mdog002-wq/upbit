import asyncio
import json
import pandas as pd
import numpy as np
import requests
import time
import os
import re
import datetime
from datetime import timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastdtw import fastdtw

# ============================================================
# 타임존 및 경로 설정
# ============================================================
KST = timezone(timedelta(hours=9))

DATA_DIR = "data"
DOCS_DIR = "docs"

INDEX_HTML_FILE = os.path.join(DOCS_DIR, "index.html")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
REMOTE_TRACKER_URL = "https://raw.githubusercontent.com/mdog002-wq/upbit/main/docs/ai_recommend_tracker.json"


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
    
    prev_close = None
    if "prev_closing_price" in df.columns:
        prev_close = df.iloc[-1]["prev_closing_price"]
    if prev_close is None or pd.isna(prev_close) or prev_close == 0:
        prev_close = df.iloc[-2]["trade_price"] if len(df) > 1 else current_price

    change_rate = ((current_price - prev_close) / prev_close) * 100

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

    liquidity_score = round(min(100.0, max(0.0, (recent_vol * current_price) / 1e8 * 2.0)), 1)
    
    high_max = df["high_price"].max()
    corpse_ratio = round(max(0.0, ((high_max - current_price) / high_max) * 100.0), 2)

    df["ma5"] = df["trade_price"].rolling(5).mean()
    df["ma20"] = df["trade_price"].rolling(20).mean()
    df["ma60"] = df["trade_price"].rolling(60).mean()
    last = df.iloc[-1]
    ma_score = 100.0 if last["ma5"] > last["ma20"] > last["ma60"] else (60.0 if last["ma5"] > last["ma20"] else 20.0)

    delta = df["trade_price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi_val = float(100 - (100 / (1 + (gain / (loss + 1e-8)).iloc[-1])))

    base_score = (
        combined_pattern_sim * weights.get("w_pattern", 0.20) +
        vol_cliff_score * weights.get("w_vol_cliff", 0.25) +
        ma_score * weights.get("w_ma_alignment", 0.25) +
        min(100.0, max(0.0, change_rate * 3.33)) * weights.get("w_daily_momentum", 0.10) +
        (current_price / high_max * 100) * weights.get("w_breakout", 0.05)
    )

    if ticker in recommended_symbols:
        base_score += 15.0

    atr = calculate_atr(df)
    tp1 = current_price + (atr * 2.0)
    tp2 = current_price + (atr * 3.5)
    sl = current_price - (atr * 1.5)

    sc = round(min(100.0, base_score), 2)
    tp1_pct = round(((tp1 - current_price) / current_price) * 100, 2)
    tp2_pct = round(((tp2 - current_price) / current_price) * 100, 2)

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
        "tp2": round(tp2, 2),
        "sl": round(sl, 2),
        "is_repo1_recommended": ticker in recommended_symbols,
        "종목명": f"{k_name} ({ticker})",
        "AI 스코어": sc,
        "현재가": current_price,
        "변동률": f"{'+' if change_rate > 0 else ''}{round(change_rate, 2)}%",
        "거래절벽": round(vol_cliff_score, 2),
        "RSI": round(rsi_val, 1),
        "유동성": liquidity_score,
        "패턴유사도": f"{combined_pattern_sim}%",
        "시체비율": f"{corpse_ratio}%",
        "저항선(1차/2차)": f"{round(tp1, 2)} / {round(tp2, 2)}",
        "목표가 1": f"{round(tp1, 2)} ({'+' if tp1_pct > 0 else ''}{tp1_pct}%)",
        "목표가 2": f"{round(tp2, 2)} ({'+' if tp2_pct > 0 else ''}{tp2_pct}%)"
    }


def generate_and_save_html(analyzed_results):
    now_str = datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

    # 테이블 행(HTML <tr>) 동적 생성
    rows_html = ""
    for item in analyzed_results:
        change_class = "text-red-500 font-bold" if "+" in item["변동률"] else "text-blue-500 font-bold"
        recommend_badge = '<span class="bg-red-100 text-red-800 text-xs font-semibold mr-2 px-2.5 py-0.5 rounded">AI추천</span>' if item["is_repo1_recommended"] else ''
        
        rows_html += f"""
        <tr class="hover:bg-gray-50 border-b">
            <td class="py-3 px-4">{recommend_badge}{item["종목명"]}</td>
            <td class="py-3 px-4 font-bold text-indigo-600">{item["AI 스코어"]}점</td>
            <td class="py-3 px-4">{format(item["현재가"], ',')}원</td>
            <td class="py-3 px-4 {change_class}">{item["변동률"]}</td>
            <td class="py-3 px-4">{item["거래절벽"]}</td>
            <td class="py-3 px-4">{item["RSI"]}</td>
            <td class="py-3 px-4">{item["유동성"]}</td>
            <td class="py-3 px-4">{item["패턴유사도"]}</td>
            <td class="py-3 px-4">{item["시체비율"]}</td>
            <td class="py-3 px-4 text-green-600">{item["목표가 1"]}</td>
            <td class="py-3 px-4 text-green-700">{item["목표가 2"]}</td>
        </tr>
        """

    # 완성형 HTML 템플릿
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 업비트 퀀트 대시보드</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 font-sans leading-normal tracking-normal">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-8 text-center">
            <h1 class="text-3xl font-bold text-gray-800">🤖 AI 업비트 퀀트 투자 대시보드</h1>
            <p class="text-gray-500 mt-2">최종 분석 시각: <span id="last-updated" class="font-semibold text-gray-700">{now_str}</span></p>
        </header>

        <div class="bg-white shadow-md rounded-lg overflow-hidden">
            <div class="overflow-x-auto">
                <table class="min-w-full bg-white border border-gray-200 text-sm text-left">
                    <thead class="bg-gray-800 text-white uppercase text-xs">
                        <tr>
                            <th class="py-3 px-4">종목명</th>
                            <th class="py-3 px-4">AI 스코어</th>
                            <th class="py-3 px-4">현재가</th>
                            <th class="py-3 px-4">변동률</th>
                            <th class="py-3 px-4">거래절벽</th>
                            <th class="py-3 px-4">RSI</th>
                            <th class="py-3 px-4">유동성</th>
                            <th class="py-3 px-4">패턴유사도</th>
                            <th class="py-3 px-4">시체비율</th>
                            <th class="py-3 px-4">목표가 1</th>
                            <th class="py-3 px-4">목표가 2</th>
                        </tr>
                    </thead>
                    <tbody class="text-gray-700">
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>
"""

    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(INDEX_HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"🕒 docs/index.html 직접 생성 및 갱신 완료: {now_str}")


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

    # HTML 파일을 곧바로 빌드하여 저장
    generate_and_save_html(analyzed_results)
    print(f"✅ 분석 및 docs/index.html 파일 갱신 완료 (1위: {analyzed_results[0]['ticker']} - {analyzed_results[0]['score']}점)")


if __name__ == "__main__":
    main()
