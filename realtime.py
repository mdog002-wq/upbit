import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
import os
import re
import threading
import time
import websockets
from fastdtw import fastdtw
import numpy as np
import pandas as pd
import requests

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
DOCS_DIR = "docs"
HTML_OUTPUT = os.path.join(DOCS_DIR, "index.html")

KST = timezone(timedelta(hours=9))


# ==========================================
# 1. AI 추천 종목 파싱
# ==========================================
def fetch_ai_recommendations():
    """upbit-a 레포지토리 docs/ai_recommend_tracker.json에서 AI 추천 티커 추출"""
    refined_set = set()

    def parse_item(item):
        if isinstance(item, str):
            match = re.search(r"\(([A-Z0-9]+)\)", item.upper())
            ticker = match.group(1) if match else item.strip().upper().replace("KRW-", "")
            if ticker and len(ticker) <= 10:
                refined_set.add(ticker)
                refined_set.add(f"KRW-{ticker}")

        elif isinstance(item, dict):
            for k, v in item.items():
                if k.isupper() and len(k) <= 10:
                    refined_set.add(k)
                    refined_set.add(f"KRW-{k}")
                
                if k in ["ticker", "symbol", "market", "code", "korean_name", "name", "coin_name"]:
                    parse_item(v)
                elif isinstance(v, (dict, list)):
                    parse_item(v)

        elif isinstance(item, list):
            for sub_item in item:
                parse_item(sub_item)

    data = None

    local_path = os.path.join("docs", "ai_recommend_tracker.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print("📁 [AI 추천] 로컬 docs/ai_recommend_tracker.json 연동 성공")
        except Exception:
            pass

    if data is None:
        url = f"https://raw.githubusercontent.com/mdog002-wq/upbit-a/main/docs/ai_recommend_tracker.json?t={int(time.time())}"
        try:
            res = requests.get(url, timeout=5, headers={"Cache-Control": "no-cache"})
            if res.status_code == 200:
                data = res.json()
                print("🌐 [AI 추천] GitHub Raw 데이터 연동 성공")
        except Exception as e:
            print(f"⚠️ AI 추천 원격 요청 실패: {e}")

    if data:
        parse_item(data)

    print(f"🤖 [AI 추천 연동 결과] 감지된 추천 티커: {sorted(list(refined_set))}")
    return refined_set


# ==========================================
# 2. DTW 계산
# ==========================================
def calculate_dtw_similarity(seq1, seq2):
    try:
        s1 = np.asarray(seq1, dtype=np.float64).reshape(-1)
        s2 = np.asarray(seq2, dtype=np.float64).reshape(-1)

        if s1.size == 0 or s2.size == 0:
            return 0.0

        min_len = min(len(s1), len(s2))
        if min_len == 0:
            return 0.0

        s1 = s1[-min_len:]
        s2 = s2[-min_len:]

        distance, _ = fastdtw(s1, s2, dist=lambda x, y: abs(x - y))
        avg_dist = distance / min_len
        similarity = np.exp(-1.5 * avg_dist) * 100.0
        return round(float(similarity), 1)
    except Exception:
        return 0.0


# ==========================================
# 3. 웹소켓 매니저
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
                                "signed_change_rate": res.get("signed_change_rate", 0) * 100,
                            }
            except Exception:
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
# 4. 유틸리티 & 데이터 수집 함수
# ==========================================
def fetch_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            markets = res.json()
            krw_markets = [m["market"] for m in markets if m["market"].startswith("KRW-")]
            market_names = {m["market"]: m["korean_name"] for m in markets if m["market"].startswith("KRW-")}
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
        btc_change = ((df.iloc[-1]["trade_price"] - df.iloc[0]["opening_price"]) / df.iloc[0]["opening_price"]) * 100
        if btc_change <= -2.0:
            return "BEAR (하락장 경고)", 0.85
        elif btc_change >= 1.5:
            return "BULL (강세장)", 1.05
        return "NEUTRAL (보통)", 1.0
    except Exception:
        return "NEUTRAL (보통)", 1.0


def calculate_historical_win_rate(history_db, target_tp_pct=5.0, target_sl_pct=2.0):
    total_trades = 0
    wins = 0
    losses = 0

    for market, records in history_db.items():
        if len(records) < 2:
            continue

        for i in range(len(records) - 1):
            entry = records[i]
            if entry.get("rank", 99) > 10:
                continue

            entry_price = entry.get("price")
            entry_ts = entry.get("timestamp")

            if not entry_price or entry_price <= 0:
                continue

            subsequent_prices = [r["price"] for r in records[i + 1 :] if r["timestamp"] > entry_ts]

            if not subsequent_prices:
                continue

            max_price = max(subsequent_prices)
            min_price = min(subsequent_prices)

            max_return = ((max_price - entry_price) / entry_price) * 100
            min_return = ((min_price - entry_price) / entry_price) * 100

            if max_return >= target_tp_pct:
                wins += 1
                total_trades += 1
            elif min_return <= -target_sl_pct:
                losses += 1
                total_trades += 1

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    return round(win_rate, 1), total_trades, wins, losses


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


# ==========================================
# 5. 개별 종목 분석 (거래량절벽 지표 적용)
# ==========================================
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

    if ws_data and "trade_price" in ws_data:
        current_price = ws_data["trade_price"]
        change_rate = ws_data["signed_change_rate"]
    else:
        current_price = df.iloc[-1]["trade_price"]
        change_rate = ((current_price - df.iloc[-1]["prev_closing_price"]) / df.iloc[-1]["prev_closing_price"]) * 100

    df_2h = df.iloc[-24:].copy().reset_index(drop=True)

    prices = df_2h["trade_price"].values
    volumes = df_2h["candle_acc_trade_volume"].values

    p_min, p_max = prices.min(), prices.max()
    v_min, v_max = volumes.min(), volumes.max()

    p_range = (p_max - p_min) if (p_max - p_min) > 1e-8 else 1.0
    v_range = (v_max - v_min) if (v_max - v_min) > 1e-8 else 1.0

    norm_prices = (prices - p_min) / p_range
    norm_volumes = (volumes - v_min) / v_range

    price_sim = calculate_dtw_similarity(norm_prices, ideal_price_pattern)
    vol_sim = calculate_dtw_similarity(norm_volumes, ideal_vol_pattern)
    combined_pattern_sim = round(price_sim * 0.7 + vol_sim * 0.3, 1)

    positive_count = sum(1 for _, row in df_2h.iterrows() if row["trade_price"] > row["opening_price"])

    volume_std = df_2h["candle_acc_trade_volume"].std()
    volume_mean = df_2h["candle_acc_trade_volume"].mean()
    ai_volatility_score = float(min(100.0, (volume_std / (volume_mean + 1e-8)) * 50))

    # -------------------------------------------------------------
    # 📉 [거래량절벽 지표 계산] 최근 거래량이 직전 평균 거래량 대비 급감한 정도 측정
    # -------------------------------------------------------------
    recent_vol_current = df.iloc[-1]["candle_acc_trade_volume"]
    avg_prev_vol = df.iloc[-21:-1]["candle_acc_trade_volume"].mean()
    
    if avg_prev_vol > 0:
        vol_ratio = recent_vol_current / avg_prev_vol
        # 거래량이 줄어들수록(절벽일수록) 높은 점수 산출
        vol_cliff_score = min(100.0, max(0.0, (1.0 - vol_ratio) * 100.0))
    else:
        vol_cliff_score = 0.0

    high_24h = df["high_price"].max()
    breakout_score = 100.0 if current_price >= high_24h else (current_price / high_24h) * 100

    recent_vol = df.iloc[-3:]["candle_acc_trade_volume"].sum()
    avg_vol = df.iloc[-36:-3]["candle_acc_trade_volume"].sum() / 33 * 3
    vol_surge_score = min(100.0, (recent_vol / (avg_vol + 1e-8)) * 25.0)

    df["ma5"] = df["trade_price"].rolling(5).mean()
    df["ma20"] = df["trade_price"].rolling(20).mean()
    df["ma60"] = df["trade_price"].rolling(60).mean()
    last_row = df.iloc[-1]

    if last_row["ma5"] > last_row["ma20"] > last_row["ma60"]:
        ma_momentum_score = 100.0
    elif last_row["ma5"] > last_row["ma20"]:
        ma_momentum_score = 60.0
    else:
        ma_momentum_score = 20.0

    up_5pct_count = sum(
        1 for _, row in df.iterrows()
        if ((row["high_price"] - row["low_price"]) / (row["low_price"] + 1e-8)) * 100 >= 5.0
        and row["trade_price"] >= row["opening_price"]
    )

    down_5pct_count = sum(
        1 for _, row in df.iterrows()
        if ((row["high_price"] - row["low_price"]) / (row["high_price"] + 1e-8)) * 100 >= 5.0
        and row["trade_price"] < row["opening_price"]
    )

    acc_24h_krw = df["candle_acc_trade_price"].sum()
    liquidity_index = (
        round(min(100.0, max(0.0, (np.log10(acc_24h_krw + 1e-8) - 7) * 20)), 1)
        if acc_24h_krw > 0 else 0.0
    )

    rsi = calculate_rsi(df["trade_price"])

    base_score = (
        combined_pattern_sim * weights.get("w_pattern", 0.10)
        + (positive_count / 24.0 * 100) * weights.get("w_buy_sell", 0.05)
        + ai_volatility_score * weights.get("w_ai_volatility", 0.05)
        + vol_cliff_score * weights.get("w_vol_cliff", 0.10)
        + breakout_score * weights.get("w_breakout", 0.15)
        + vol_surge_score * weights.get("w_vol_surge", 0.20)
        + ma_momentum_score * weights.get("w_ma_alignment", 0.10)
        + min(100.0, max(0.0, change_rate * 3.33)) * weights.get("w_daily_momentum", 0.25)
    )

    final_score = max(0.0, base_score * btc_multiplier)

    if is_ai_recommended:
        if rsi < 68.0 and vol_surge_score >= 10.0:
            final_score *= 1.015
        elif liquidity_index < 10.0 or combined_pattern_sim < 30.0:
            final_score *= 0.95

    return {
        "market": market,
        "ticker": ticker,
        "name": k_name,
        "current_price": current_price,
        "change_rate": round(change_rate, 2),
        "pattern_similarity": combined_pattern_sim,
        "positive_count": positive_count,
        "vol_cliff_score": round(vol_cliff_score, 1),
        "score": round(final_score, 2),
        "rsi": round(rsi, 1),
        "ai_volatility_score": round(ai_volatility_score, 1),
        "up_5pct_count": up_5pct_count,
        "down_5pct_count": down_5pct_count,
        "liquidity_index": liquidity_index,
        "is_ai_recommended": is_ai_recommended,
    }


# ==========================================
# 6. HTML 생성
# ==========================================
def generate_full_dashboard_html(
    analysis_results,
    current_time_str,
    btc_status,
    backtest_stats,
    html_path=HTML_OUTPUT,
):
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    win_rate, total_trades, wins, losses = backtest_stats

    rows_list = []
    for item in analysis_results:
        change_class = "plus" if item["change_rate"] > 0 else ("minus" if item["change_rate"] < 0 else "")
        change_sign = "+" if item["change_rate"] > 0 else ""

        ai_badge_html = '<span class="ai-badge">AI추천</span>' if item.get("is_ai_recommended") else ""

        rsi_display = f"{item['rsi']}"
        if item["rsi"] >= 70:
            rsi_display = f"<span class='overheat'>{item['rsi']} (과열)</span>"

        row = f"""
<tr>
<td><b>{item['rank']}</b></td>
<td>
    <a href="#" onclick="openChartModal('{item['ticker']}', '{item['name']}'); return false;" class="coin-link">
        <b>{item['name']}</b> <span class="ticker-symbol">({item['ticker']})</span>
    </a>{ai_badge_html}
</td>
<td>{item['current_price']:,}</td>
<td class="{change_class}">{change_sign}{item['change_rate']}%</td>
<td>{rsi_display}</td>
<td><b>{item['pattern_similarity']}%</b></td>
<td class="vol-cliff">{item['vol_cliff_score']}점</td>
<td class="liquidity">{item['liquidity_index']}점</td>
<td><b>{item['score']}점</b></td>
<td><span class="plus">▲ {item['up_5pct_count']}회</span> / <span class="minus">▼ {item['down_5pct_count']}회</span></td>
</tr>"""
        rows_list.append(row)

    rows_html = "".join(rows_list)

    html_template = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>업비트 실시간 급등주 포착 대시보드</title>
  <meta http-equiv="refresh" content="300">
<style>
body { background-color: #f8f9fa; color: #333333; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; }
.header-container { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; background: #ffffff; padding: 15px 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 15px; }
.header-left { text-align: left; } .header-center { text-align: center; } .header-right { text-align: right; font-size: 13px; color: #495057; font-weight: 500; }
.ai-btn { background-color: #007bff; color: white; padding: 10px 18px; border-radius: 5px; text-decoration: none; font-weight: bold; font-size: 14px; display: inline-block; transition: background 0.2s; }
.ai-btn:hover { background-color: #0056b3; }

.status-card {
    background: #ffffff;
    padding: 12px 20px;
    border-radius: 8px;
    margin-bottom: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    font-size: 14px;
    font-weight: bold;
    color: #495057;
}

.winrate-card { border-left: 6px solid #2b8a3e; }
.winrate-val { font-size: 18px; color: #e03131; }

.ai-badge {
    background-color: #e03131 !important;
    color: #ffffff !important;
    font-size: 11px !important;
    font-weight: bold !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    margin-left: 6px !important;
    display: inline-block !important;
    vertical-align: middle !important;
}

.coin-link { color: #333333; text-decoration: none; cursor: pointer; }
.coin-link:hover { color: #007bff; text-decoration: underline; }

.search-box { margin-bottom: 20px; }
.search-box input { width: 100%; padding: 12px 15px; font-size: 16px; border: 1px solid #ced4da; border-radius: 6px; box-sizing: border-box; outline: none; background: #ffffff; }

.table-container { max-height: 75vh; overflow-y: auto; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); background: #ffffff; }
table { width: 100%; border-collapse: collapse; background: #ffffff; }
th, td { padding: 12px 15px; text-align: center; border-bottom: 1px solid #e9ecef; }
th { 
    position: sticky; 
    top: 0; 
    z-index: 10; 
    background-color: #f1f3f5; 
    color: #495057; 
    font-weight: 600; 
    cursor: pointer; 
    user-select: none; 
    transition: background-color 0.2s; 
    box-shadow: inset 0 -1px 0 #e9ecef;
}
th:hover { background-color: #e9ecef; }
tbody tr { transition: background-color 0.15s; }
tbody tr:hover { background-color: #e9ecef !important; }
.plus { color: #e03131; font-weight: bold; }
.minus { color: #1971c2; font-weight: bold; }
.overheat { color: #d9480f; font-weight: bold; }
.ticker-symbol { font-size: 12px; color: #868e96; font-weight: normal; margin-left: 4px; }
.vol-cliff { color: #d9480f; font-weight: bold; }
.liquidity { color: #2b8a3e; font-weight: bold; }

.modal-overlay {
    display: none;
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0, 0, 0, 0.6);
    z-index: 9999;
    justify-content: center;
    align-items: center;
}
.modal-content {
    background: #ffffff;
    width: 90%;
    max-width: 1000px;
    height: 650px;
    border-radius: 12px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.modal-header {
    padding: 15px 20px;
    background: #1e222d;
    color: #ffffff;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.modal-title { font-size: 18px; font-weight: bold; }
.modal-close { font-size: 24px; cursor: pointer; color: #cccccc; line-height: 1; }
.modal-close:hover { color: #ffffff; }
.modal-body { flex: 1; width: 100%; height: 100%; background: #131722; }
</style>
<script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
<script>
function filterTable() {
    let input = document.getElementById('searchInput').value.toLowerCase();
    let tr = document.getElementById('coinTable').getElementsByTagName('tr');
    for (let i = 1; i < tr.length; i++) {
        let td = tr[i].getElementsByTagName('td')[1];
        tr[i].style.display = (td && (td.textContent || td.innerText).toLowerCase().indexOf(input) > -1) ? "" : "none";
    }
}

let sortDirections = {};
function sortTable(columnIndex) {
    const table = document.getElementById("coinTable");
    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));

    const isAscending = sortDirections[columnIndex] === true;
    sortDirections[columnIndex] = !isAscending;

    rows.sort((rowA, rowB) => {
        let cellA = rowA.children[columnIndex].textContent.trim();
        let cellB = rowB.children[columnIndex].textContent.trim();

        let numA = parseFloat(cellA.replace(/[^0-9.-]+/g, ""));
        let numB = parseFloat(cellB.replace(/[^0-9.-]+/g, ""));

        if (!isNaN(numA) && !isNaN(numB)) {
            return isAscending ? numA - numB : numB - numA;
        } else {
            return isAscending 
                ? cellA.localeCompare(cellB, 'ko-KR') 
                : cellB.localeCompare(cellA, 'ko-KR');
        }
    });

    rows.forEach(row => tbody.appendChild(row));
}

function openChartModal(ticker, name) {
    document.getElementById('modalTitle').innerText = name + ' (' + ticker + ') 실시간 차트';
    document.getElementById('chartModal').style.display = 'flex';
    document.getElementById('tvChartContainer').innerHTML = '';
    
    new TradingView.widget({
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
    });
}

function closeChartModal() {
    document.getElementById('chartModal').style.display = 'none';
    document.getElementById('tvChartContainer').innerHTML = '';
}

window.onkeydown = function(event) {
    if (event.keyCode === 27) closeChartModal();
};
</script>
</head>
<body>

<div class="header-container">
<div class="header-left"><a href="https://upbit-a.onrender.com" target="_self" class="ai-btn">AI리포트이동</a></div>
<div class="header-center"><h2 style="margin: 0; font-size: 20px;">🚀 실시간 DTW + 웹소켓 고도화 대시보드</h2></div>
<div class="header-right">마지막 업데이트: <b>{{CURRENT_TIME}}</b></div>
</div>

<div class="status-card winrate-card">
🎯 <b>실시간 백테스팅 승률 (익절 +5% / 손절 -2% 기준):</b> 
<span class="winrate-val">{{WIN_RATE}}%</span> 
<span style="font-size: 13px; color: #666; font-weight: normal;">(최근 포착 TOP10 종목 총 {{TOTAL_TRADES}}건 검증 — {{WINS}}승 {{LOSSES}}패)</span>
</div>

<div class="status-card">
🌐 비트코인(BTC) 시장 상황: <span style="color:#007bff;">{{BTC_STATUS}}</span> (약세장 감점 적용 여부 판별)
</div>

<div class="search-box"><input type="text" id="searchInput" onkeyup="filterTable()" placeholder="코인명 또는 티커 검색..."></div>

<div class="table-container">
<table id="coinTable">
<thead>
<tr>
<th onclick="sortTable(0)">순위</th>
<th onclick="sortTable(1)">한글코인명</th>
<th onclick="sortTable(2)">현재가격 (KRW)</th>
<th onclick="sortTable(3)">전일대비 등락률</th>
<th onclick="sortTable(4)">RSI(14)</th>
<th onclick="sortTable(5)">DTW패턴유사도</th>
<th onclick="sortTable(6)">거래량절벽</th>
<th onclick="sortTable(7)">유동성</th>
<th onclick="sortTable(8)">최종예측점수</th>
<th onclick="sortTable(9)">5% 변동 (상승/하락)</th>
</tr>
</thead>
<tbody>
{{ROWS}}
</tbody>
</table>
</div>

<div id="chartModal" class="modal-overlay" onclick="if(event.target === this) closeChartModal();">
    <div class="modal-content">
        <div class="modal-header">
            <span id="modalTitle" class="modal-title">코인 차트</span>
            <span class="modal-close" onclick="closeChartModal()">&times;</span>
        </div>
        <div class="modal-body" id="tvChartContainer"></div>
    </div>
</div>

</body>
</html>
"""

    final_html = (
        html_template.replace("{{CURRENT_TIME}}", current_time_str)
        .replace("{{BTC_STATUS}}", btc_status)
        .replace("{{WIN_RATE}}", str(win_rate))
        .replace("{{TOTAL_TRADES}}", str(total_trades))
        .replace("{{WINS}}", str(wins))
        .replace("{{LOSSES}}", str(losses))
        .replace("{{ROWS}}", rows_html)
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)


# ==========================================
# 7. 메인 실행 함수
# ==========================================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    krw_markets, market_names = fetch_krw_markets()

    ws_manager = UpbitWebSocketManager(krw_markets)
    ws_manager.start()
    time.sleep(2)

    ai_recommend_set = fetch_ai_recommendations()
    btc_status, btc_multiplier = check_btc_status()

    history_db = load_json(HISTORY_FILE, {})
    backtest_stats = calculate_historical_win_rate(
        history_db, target_tp_pct=5.0, target_sl_pct=2.0
    )

    weights = load_json(
        WEIGHTS_FILE,
        {
            "w_pattern": 0.10,
            "w_buy_sell": 0.05,
            "w_ai_volatility": 0.05,
            "w_vol_cliff": 0.10,
            "w_breakout": 0.15,
            "w_vol_surge": 0.20,
            "w_ma_alignment": 0.10,
            "w_daily_momentum": 0.25,
        },
    )

    pattern_data = load_json(PATTERN_FILE, {})
    raw_price = pattern_data.get("golden_pattern", np.linspace(0.2, 1.0, 24).tolist())
    raw_vol = pattern_data.get("golden_volume_pattern", np.linspace(0.1, 1.0, 24).tolist())

    ideal_price_pattern = np.squeeze(np.asarray(raw_price, dtype=np.float64)).flatten()
    ideal_vol_pattern = np.squeeze(np.asarray(raw_vol, dtype=np.float64)).flatten()

    analysis_results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                analyze_single_coin,
                market,
                market_names.get(market, market),
                ideal_price_pattern,
                ideal_vol_pattern,
                history_db,
                weights,
                btc_multiplier,
                ws_manager.ticker_data.get(market, {}),
                (
                    market in ai_recommend_set
                    or market.replace("KRW-", "") in ai_recommend_set
                ),
            ): market
            for market in krw_markets
        }

        for future in as_completed(futures):
            res = future.result()
            if res:
                analysis_results.append(res)

    ws_manager.stop()

    # 최종 점수 기준 내림차순 정렬
    analysis_results.sort(key=lambda x: x["score"], reverse=True)

    for idx, item in enumerate(analysis_results):
        rank = idx + 1
        item["rank"] = rank
        m_code = item["market"]
        if m_code not in history_db:
            history_db[m_code] = []
        history_db[m_code].append(
            {
                "timestamp": time.time(),
                "score": item["score"],
                "rank": rank,
                "price": item["current_price"],
            }
        )
        history_db[m_code] = [
            h for h in history_db[m_code] if h["timestamp"] >= time.time() - 86400
        ][-50:]

    save_json(HISTORY_FILE, history_db)
    save_json(WEIGHTS_FILE, weights)

    current_time_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    generate_full_dashboard_html(
        analysis_results,
        current_time_str,
        btc_status,
        backtest_stats,
        HTML_OUTPUT,
    )
    print("🎨 [대시보드] 거래량절벽 지표 적용 및 대시보드 생성 완료!")


if __name__ == "__main__":
    main()
