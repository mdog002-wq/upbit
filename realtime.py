import asyncio
import json
import websockets
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
HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
REMOTE_TRACKER_URL = "https://raw.githubusercontent.com/mdog002-wq/upbit/main/docs/ai_recommend_tracker.json"
WARNING_COINS_URL = "https://raw.githubusercontent.com/mdog002-wq/upbit-a/main/docs/warning_coins.json"


def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # history_db.json 파일이 리스트 형태인 경우 에러 방지를 위해 default 반환
                if filepath == HISTORY_FILE and not isinstance(data, dict):
                    return default
                return data
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


class UpbitWebSocketManager:
    def __init__(self, markets):
        self.markets = markets
        self.ticker_data = {}
        self.is_running = False

    async def _connect_websocket(self):
        url = "wss://api.upbit.com/websocket/v1"
        subscribe_fmt = [{"ticket": "TEST_TICKET"}, {"type": "ticker", "codes": self.markets}]
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
                                "signed_change_rate": res.get("signed_change_rate", 0) * 100,
                                "prev_closing_price": res.get("prev_closing_price")
                            }
            except Exception:
                await asyncio.sleep(2)

    def start(self):
        self.is_running = True
        import threading
        t = threading.Thread(target=lambda: asyncio.run(self._connect_websocket()), daemon=True)
        t.start()

    def stop(self):
        self.is_running = False


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
    recommended_set = set()
    try:
        res = requests.get(f"{REMOTE_TRACKER_URL}?t={int(time.time())}", timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data and isinstance(data, list):
                latest = data[-1]
                for c in latest.get("recommended_coins", []):
                    sym = c.get("symbol")
                    if sym:
                        clean_sym = sym.replace("KRW-", "").upper()
                        recommended_set.add(clean_sym)
                        recommended_set.add(f"KRW-{clean_sym}")
    except Exception:
        pass
    return recommended_set


def fetch_warning_coins():
    warning_set = set()
    try:
        res = requests.get(f"{WARNING_COINS_URL}?t={int(time.time())}", timeout=5)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "symbol" in item:
                        symbol = item["symbol"].strip().upper().replace("KRW-", "")
                        warning_set.add(symbol)
                        warning_set.add(f"KRW-{symbol}")
                    elif isinstance(item, str):
                        symbol = item.strip().upper().replace("KRW-", "")
                        warning_set.add(symbol)
                        warning_set.add(f"KRW-{symbol}")
    except Exception:
        pass
    return warning_set


def check_btc_status():
    try:
        btc_candles = fetch_5m_candles("KRW-BTC", count=24)
        if len(btc_candles) < 24: 
            return "NEUTRAL (보통)", 1.0
        df = pd.DataFrame(btc_candles).sort_values("timestamp")
        btc_change = ((df.iloc[-1]["trade_price"] - df.iloc[0]["opening_price"]) / df.iloc[0]["opening_price"]) * 100
        if btc_change <= -2.0: 
            return "BEAR (하락장 경고)", 0.85
        elif btc_change >= 1.5: 
            return "BULL (강세장)", 1.05
        return "NEUTRAL (보통)", 1.0
    except Exception:
        return "NEUTRAL (보통)", 1.0


def calculate_historical_win_rate(history_db, target_tp_pct=5.0, target_sl_pct=2.0):
    if not isinstance(history_db, dict):
        return 0.0, 0, 0, 0
    total_trades, wins, losses = 0, 0, 0
    for market, records in history_db.items():
        if not isinstance(records, list) or len(records) < 2: 
            continue
        for i in range(len(records) - 1):
            entry = records[i]
            if not isinstance(entry, dict) or entry.get("rank", 99) > 10: 
                continue
            entry_price, entry_ts = entry.get("price"), entry.get("timestamp")
            if not entry_price or entry_price <= 0: 
                continue
            subsequent_prices = [r["price"] for r in records[i+1:] if isinstance(r, dict) and "price" in r and r.get("timestamp", 0) > entry_ts]
            if not subsequent_prices: 
                continue
            
            max_return = ((max(subsequent_prices) - entry_price) / entry_price) * 100
            min_return = ((min(subsequent_prices) - entry_price) / entry_price) * 100

            if max_return >= target_tp_pct:
                wins += 1
                total_trades += 1
            elif min_return <= -target_sl_pct:
                losses += 1
                total_trades += 1

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    return round(win_rate, 1), total_trades, wins, losses


def analyze_single_coin(market, k_name, golden_price_patterns, golden_vol_patterns, weights, recommended_symbols, warning_symbols, btc_multiplier, ws_data):
    ticker = market.replace("KRW-", "")
    candles = fetch_5m_candles(market, count=120)
    if len(candles) < 60:
        return None

    df = pd.DataFrame(candles).sort_values("timestamp").reset_index(drop=True)
    
    if ws_data and "trade_price" in ws_data and ws_data["trade_price"]:
        current_price = ws_data["trade_price"]
        change_rate = ws_data["signed_change_rate"]
    else:
        current_price = df.iloc[-1]["trade_price"]
        prev_close = df.iloc[-1].get("prev_closing_price")
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

    is_ai_rec = (ticker in recommended_symbols or market in recommended_symbols)
    is_warn = (ticker in warning_symbols or market in warning_symbols)

    if is_ai_rec:
        base_score += 15.0

    final_score = max(0.0, base_score * btc_multiplier)
    if is_warn:
        final_score *= 0.85

    atr = calculate_atr(df)
    tp1 = current_price + (atr * 2.0)
    tp2 = current_price + (atr * 3.5)
    sl = current_price - (atr * 1.5)

    sc = round(min(100.0, final_score), 2)
    tp1_pct = round(((tp1 - current_price) / current_price) * 100, 2)
    tp2_pct = round(((tp2 - current_price) / current_price) * 100, 2)
    sl_pct = round(((sl - current_price) / current_price) * 100, 2)

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
        "tp1_pct": tp1_pct,
        "tp2_pct": tp2_pct,
        "sl_pct": sl_pct,
        "is_ai_recommended": is_ai_rec,
        "is_warning": is_warn,
        "종목명": f"{k_name} ({ticker})",
        "AI 스코어": sc,
        "현재가": current_price,
        "변동률": f"{'+' if change_rate > 0 else ''}{round(change_rate, 2)}%",
        "거래절벽": round(vol_cliff_score, 2),
        "RSI": round(rsi_val, 1),
        "유동성": liquidity_score,
        "패턴유사도": f"{combined_pattern_sim}%",
        "시체비율": f"{corpse_ratio}%"
    }


def generate_and_save_html(analyzed_results, current_time_str, btc_status, backtest_stats):
    win_rate, total_trades, wins, losses = backtest_stats
    dashboard_json_data = json.dumps(analyzed_results, ensure_ascii=False)

    rows_html = ""
    for item in analyzed_results:
        change_class = "text-red-500 font-bold" if item["change_rate"] > 0 else ("text-blue-500 font-bold" if item["change_rate"] < 0 else "")
        change_sign = "+" if item["change_rate"] > 0 else ""
        
        ai_badge = '<span class="bg-blue-600 text-white text-xs font-bold px-2 py-0.5 rounded ml-1">AI추천</span>' if item.get("is_ai_recommended") else ''
        warning_badge = '<span class="bg-red-500 text-white text-xs font-bold px-2 py-0.5 rounded ml-1 animate-pulse">⚠️ 위험</span>' if item.get("is_warning") else ''
        rsi_display = f"{item['rsi']}" if item["rsi"] < 70 else f"<span class='text-orange-600 font-bold'>{item['rsi']} (과열)</span>"

        rows_html += f"""
        <tr class="hover:bg-gray-50 border-b transition-colors">
            <td class="py-3 px-4 font-bold text-gray-700">{item['rank']}</td>
            <td class="py-3 px-4">
                <a href="#" onclick="openChartModal('{item['ticker']}', '{item['name']}'); return false;" class="text-gray-900 font-semibold hover:text-blue-600 underline">
                    {item['name']} <span class="text-xs text-gray-400 font-normal">({item['ticker']})</span>
                </a>{ai_badge}{warning_badge}
            </td>
            <td class="py-3 px-4 font-medium">{format(item['current_price'], ',')}원</td>
            <td class="py-3 px-4 {change_class}">{change_sign}{item['change_rate']}%</td>
            <td class="py-3 px-4">{rsi_display}</td>
            <td class="py-3 px-4 font-bold text-indigo-600">{item['pattern_similarity']}%</td>
            <td class="py-3 px-4 text-orange-500 font-bold">{item['거래절벽']}점</td>
            <td class="py-3 px-4 text-green-600 font-bold">{item['유동성']}점</td>
            <td class="py-3 px-4 font-bold text-purple-600 text-base">{item['score']}점</td>
            <td class="py-3 px-4 text-green-600 font-semibold">{item['tp1']} ({'+' if item['tp1_pct'] > 0 else ''}{item['tp1_pct']}%)</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 업비트 퀀트 투자 대시보드</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <style>
        .modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.6); z-index: 9999; justify-content: center; align-items: center; }}
        .gemini-fab {{ position: fixed; bottom: 25px; right: 25px; background: linear-gradient(135deg, #1a73e8, #8ab4f8); color: #ffffff; padding: 12px 20px; border-radius: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.25); cursor: pointer; font-weight: bold; font-size: 15px; display: flex; align-items: center; gap: 8px; z-index: 999; transition: transform 0.2s, box-shadow 0.2s; }}
        .gemini-fab:hover {{ transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.35); }}
        .gemini-chat-container {{ display: none; position: fixed; bottom: 85px; right: 25px; width: 380px; height: 520px; background: #ffffff; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.25); z-index: 1000; flex-direction: column; overflow: hidden; border: 1px solid #e0e0e0; }}
    </style>
    <script>
        const dashboardData = {dashboard_json_data};

        function filterTable() {{
            let input = document.getElementById('searchInput').value.toLowerCase();
            let tr = document.getElementById('coinTable').getElementsByTagName('tr');
            for (let i = 1; i < tr.length; i++) {{
                let td = tr[i].getElementsByTagName('td')[1];
                tr[i].style.display = (td && (td.textContent || td.innerText).toLowerCase().indexOf(input) > -1) ? "" : "none";
            }}
        }}

        function runAiDiagnosis() {{
            const inputKeyword = document.getElementById('aiCoinInput').value.trim().toUpperCase();
            const entryPrice = parseFloat(document.getElementById('aiPriceInput').value);
            const resultCard = document.getElementById('aiResultCard');

            if (!inputKeyword || isNaN(entryPrice) || entryPrice <= 0) {{
                alert('올바른 종목명/티커와 진입가를 입력해주세요.');
                return;
            }}

            const coin = dashboardData.find(item => 
                item.ticker.toUpperCase() === inputKeyword || 
                item.name.toUpperCase() === inputKeyword || 
                item.market.toUpperCase() === 'KRW-' + inputKeyword
            );

            if (!coin) {{
                alert('대시보드 리스트에서 해당 코인을 찾을 수 없습니다.');
                return;
            }}

            const ratio = entryPrice / coin.current_price;
            const tp1 = coin.tp1 * ratio;
            const tp2 = coin.tp2 * ratio;
            const sl = coin.sl * ratio;

            let comment = `📊 예측 점수 <strong>${{coin.score}}점</strong>, DTW 유사도 <strong>${{coin.pattern_similarity}}%</strong> 차트입니다.<br>`;
            if (coin.is_warning) {{
                comment += `🚨 <strong>위험 코인 경고:</strong> 이 코인은 위험 코인 목록(warning_coins.json)에 지정되어 있으므로 변동성에 각별히 주의하세요!<br>`;
            }}

            resultCard.style.display = 'block';
            resultCard.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <strong style="font-size: 16px; color: #1e222d;">🤖 [${{coin.name}} / ${{coin.ticker}}] AI 스마트 진단 리포트</strong>
                        <span style="font-size: 13px; color: #555; margin-left: 8px;">(기준가: ${{coin.current_price.toLocaleString()}} KRW)</span>
                    </div>
                    <div style="font-size: 13px; display: flex; gap: 10px; flex-wrap: wrap;">
                        <span style="color: #2b8a3e; font-weight: bold;">🎯 1차: ${{tp1.toLocaleString(undefined, {{maximumFractionDigits: 2}})}} (${{coin.tp1_pct}}%)</span>
                        <span style="color: #2b8a3e; font-weight: bold;">🎯 2차: ${{tp2.toLocaleString(undefined, {{maximumFractionDigits: 2}})}} (${{coin.tp2_pct}}%)</span>
                        <span style="color: #e03131; font-weight: bold;">🛑 손절가: ${{sl.toLocaleString(undefined, {{maximumFractionDigits: 2}})}} (${{coin.sl_pct}}%)</span>
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 13px; color: #333; background: #ffffff; padding: 8px 12px; border-radius: 4px; border-left: 4px solid #007bff;">
                    ${{comment}}
                </div>
            `;
        }}

        function openChartModal(ticker, name) {{
            document.getElementById('modalTitle').innerText = name + ' (' + ticker + ') 실시간 차트';
            document.getElementById('chartModal').style.display = 'flex';
            document.getElementById('tvChartContainer').innerHTML = '';
            
            new TradingView.widget({{
                "autosize": true,
                "symbol": "UPBIT:" + ticker + "KRW",
                "interval": "5",
                "timezone": "Asia/Seoul",
                "theme": "dark",
                "style": "1",
                "locale": "kr",
                "toolbar_bg": "#f1f3f5",
                "enable_publishing": false,
                "allow_symbol_change": true,
                "container_id": "tvChartContainer"
            }});
        }}

        function closeChartModal() {{
            document.getElementById('chartModal').style.display = 'none';
            document.getElementById('tvChartContainer').innerHTML = '';
        }}

        function toggleGeminiChat() {{
            const chatBox = document.getElementById('geminiChatContainer');
            chatBox.style.display = (chatBox.style.display === 'none' || chatBox.style.display === '') ? 'flex' : 'none';
        }}
    </script>
</head>
<body class="bg-gray-100 font-sans leading-normal tracking-normal p-4">
    <div class="container mx-auto max-w-7xl">
        <header class="bg-white p-6 rounded-lg shadow-md mb-6 flex flex-col md:flex-row justify-between items-center gap-4">
            <div>
                <h1 class="text-2xl font-bold text-gray-800">🤖 AI 업비트 퀀트 투자 대시보드</h1>
                <p class="text-gray-500 text-sm mt-1">최종 분석 시각: <span class="font-semibold text-gray-700">{current_time_str}</span></p>
            </div>
            <div class="flex items-center gap-3">
                <span class="bg-blue-50 text-blue-700 px-3 py-1.5 rounded-lg text-sm font-semibold border border-blue-200">🌐 비트코인 시장: {btc_status}</span>
            </div>
        </header>

        <div class="bg-white p-4 rounded-lg shadow-md mb-6 border-l-4 border-green-600 text-sm font-medium text-gray-700">
            🎯 <b>실시간 백테스팅 승률 (익절 +5% / 손절 -2% 기준):</b> 
            <span class="text-red-600 text-base font-bold">{win_rate}%</span> 
            <span class="text-gray-500 font-normal">(최근 포착 TOP10 종목 총 {total_trades}건 검증 — {wins}승 {losses}패)</span>
        </div>

        <div class="flex flex-col md:flex-row justify-between items-center gap-4 mb-6">
            <div class="w-full md:w-1/3">
                <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="코인명 또는 티커 검색..." class="w-full px-4 py-2 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-blue-500 bg-white">
            </div>
            <div class="w-full md:w-auto flex items-center gap-2 bg-white p-2 rounded-lg border border-gray-300 shadow-sm">
                <span class="font-bold text-xs text-blue-600">🤖 AI 스마트 진단:</span>
                <input type="text" id="aiCoinInput" placeholder="종목명/티커" class="px-2 py-1 border border-gray-300 rounded text-sm w-24 outline-none">
                <input type="number" id="aiPriceInput" placeholder="진입가 (KRW)" class="px-2 py-1 border border-gray-300 rounded text-sm w-28 outline-none">
                <button onclick="runAiDiagnosis()" class="bg-green-600 hover:bg-green-700 text-white font-bold px-3 py-1 rounded text-sm transition-colors">분석하기</button>
            </div>
        </div>

        <div id="aiResultCard" class="hidden bg-blue-50 border border-blue-200 p-4 rounded-lg mb-6 shadow-sm"></div>

        <div class="bg-white shadow-md rounded-lg overflow-hidden">
            <div class="overflow-x-auto max-h-[70vh]">
                <table id="coinTable" class="min-w-full bg-white border border-gray-200 text-sm text-center">
                    <thead class="bg-gray-800 text-white uppercase text-xs sticky top-0 z-10">
                        <tr>
                            <th class="py-3 px-4">순위</th>
                            <th class="py-3 px-4">한글코인명</th>
                            <th class="py-3 px-4">현재가격 (KRW)</th>
                            <th class="py-3 px-4">전일대비 등락률</th>
                            <th class="py-3 px-4">RSI(14)</th>
                            <th class="py-3 px-4">DTW패턴유사도</th>
                            <th class="py-3 px-4">거래량절벽</th>
                            <th class="py-3 px-4">유동성</th>
                            <th class="py-3 px-4">최종예측점수</th>
                            <th class="py-3 px-4">목표가 1차</th>
                        </tr>
                    </thead>
                    <tbody class="text-gray-700">
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div id="chartModal" class="modal-overlay" onclick="if(event.target === this) closeChartModal();">
        <div class="bg-white w-[90%] max-w-4xl h-[650px] rounded-xl shadow-2xl flex flex-col overflow-hidden">
            <div class="bg-gray-900 text-white px-5 py-4 flex justify-between items-center">
                <span id="modalTitle" class="font-bold text-lg">코인 차트</span>
                <span class="text-2xl cursor-pointer text-gray-400 hover:text-white" onclick="closeChartModal()">&times;</span>
            </div>
            <div class="flex-1 w-full h-full bg-[#131722]" id="tvChartContainer"></div>
        </div>
    </div>

    <div class="gemini-fab" onclick="toggleGeminiChat()">
        🤖 Gemini AI
    </div>

    <div id="geminiChatContainer" class="gemini-chat-container">
        <div class="bg-blue-600 text-white px-4 py-3 font-bold text-sm flex justify-between items-center">
            <span>✨ Gemini AI Assistant</span>
            <span class="cursor-pointer text-lg" onclick="toggleGeminiChat()">&times;</span>
        </div>
        <iframe class="w-full h-full border-none" src="https://gemini.google.com/"></iframe>
    </div>
</body>
</html>
"""

    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(INDEX_HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    weights = load_json(WEIGHTS_FILE, {
        "w_pattern": 0.20, "w_vol_cliff": 0.25, "w_ma_alignment": 0.25,
        "w_vol_surge": 0.15, "w_daily_momentum": 0.10, "w_breakout": 0.05
    })

    pattern_data = load_json(PATTERN_FILE, {})
    golden_price_patterns = pattern_data.get("golden_patterns", [])
    golden_vol_patterns = pattern_data.get("golden_volume_patterns", [])

    recommended_symbols = fetch_remote_recommendations()
    warning_symbols = fetch_warning_coins()
    btc_status, btc_multiplier = check_btc_status()

    history_db = load_json(HISTORY_FILE, {})
    if not isinstance(history_db, dict):
        history_db = {}
        
    backtest_stats = calculate_historical_win_rate(history_db, target_tp_pct=5.0, target_sl_pct=2.0)

    res = requests.get("https://api.upbit.com/v1/market/all")
    all_krw = [m["market"] for m in res.json() if m["market"].startswith("KRW-")]
    market_names = {m["market"]: m["korean_name"] for m in res.json() if m["market"].startswith("KRW-")}

    ws_manager = UpbitWebSocketManager(all_krw)
    ws_manager.start()
    time.sleep(2)

    analyzed_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                analyze_single_coin,
                market,
                market_names.get(market, market),
                golden_price_patterns,
                golden_vol_patterns,
                weights,
                recommended_symbols,
                warning_symbols,
                btc_multiplier,
                ws_manager.ticker_data.get(market, {})
            ) for market in all_krw
        ]
        for f in as_completed(futures):
            r = f.result()
            if r:
                analyzed_results.append(r)

    ws_manager.stop()
    analyzed_results.sort(key=lambda x: x["score"], reverse=True)

    for idx, item in enumerate(analyzed_results):
        rank = idx + 1
        item["rank"] = rank
        m_code = item["market"]
        if m_code not in history_db:
            history_db[m_code] = []
        history_db[m_code].append({
            "timestamp": time.time(), "score": item["score"], "rank": rank, "price": item["current_price"]
        })
        history_db[m_code] = [h for h in history_db[m_code] if isinstance(h, dict) and h.get("timestamp", 0) >= time.time() - 86400][-50:]

    save_json(HISTORY_FILE, history_db)
    save_json(WEIGHTS_FILE, weights)

    current_time_str = datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    generate_and_save_html(analyzed_results, current_time_str, btc_status, backtest_stats)
    print(f"✅ 분석 및 docs/index.html 파일 갱신 완료 (1위: {analyzed_results[0]['ticker']} - {analyzed_results[0]['score']}점)")


if __name__ == "__main__":
    main()
