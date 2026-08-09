from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
import os
import time
import numpy as np
import pandas as pd
import requests

# 경로 및 상수 설정
DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
DOCS_DIR = "docs"
HTML_OUTPUT = os.path.join(DOCS_DIR, "index.html")

KST = timezone(timedelta(hours=9))


def fetch_ai_recommendations():
    """GitHub 저장소의 ai_recommend_tracker.json 파싱"""
    url = "https://raw.githubusercontent.com/mdog002-wq/upbit-a/main/docs/ai_recommend_tracker.json"
    refined_set = set()
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            raw_tickers = []
            
            if isinstance(data, dict):
                if "recommended_tickers" in data:
                    raw_tickers = data.get("recommended_tickers", [])
                elif "recommended_coins" in data:
                    raw_tickers = data.get("recommended_coins", [])
                else:
                    raw_tickers = list(data.keys())
            elif isinstance(data, list):
                raw_tickers = data

            for t in raw_tickers:
                if not t or not isinstance(t, str):
                    continue
                t_str = t.strip().upper()
                refined_set.add(t_str)
                refined_set.add(t_str.replace("KRW-", ""))
                refined_set.add(f"KRW-{t_str.replace('KRW-', '')}")

            return refined_set
    except Exception as e:
        print(f"⚠️ [AI 추천 연동] JSON 불러오기 실패: {e}")

    return set()


def fetch_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    res = requests.get(url)
    if res.status_code != 200:
        return [], {}
    markets = res.json()
    krw_markets = [m["market"] for m in markets if m["market"].startswith("KRW-")]
    market_names = {
        m["market"]: m["korean_name"]
        for m in markets
        if m["market"].startswith("KRW-")
    }
    return krw_markets, market_names


def fetch_ticker_data(markets):
    """업비트 원화 마켓 전일 대비 등락률 일괄 수집"""
    markets_str = ",".join(markets)
    url = f"https://api.upbit.com/v1/ticker?markets={markets_str}"
    ticker_dict = {}
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            for item in data:
                ticker_dict[item["market"]] = item["signed_change_rate"] * 100
    except Exception as e:
        print(f"⚠️ [Ticker 연동 실패] 전일대비 데이터 수집 오류: {e}")
    return ticker_dict


def fetch_candles(market, count=672):
    all_candles = []
    to_param = ""
    while len(all_candles) < count:
        req_count = min(200, count - len(all_candles))
        url = f"https://api.upbit.com/v1/candles/minutes/15?market={market}&count={req_count}"
        if to_param:
            url += f"&to={to_param}"

        try:
            res = requests.get(url, timeout=5)
            if res.status_code != 200:
                time.sleep(0.2)
                continue
            data = res.json()
            if not data:
                break
            all_candles.extend(data)
            if len(data) < req_count:
                break
            to_param = data[-1]["candle_date_time_utc"]
        except Exception:
            time.sleep(0.2)
            continue

    return all_candles


def calculate_rsi(series, period=14):
    """RSI(상대강도지수) 계산"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50.0


def check_btc_market_status():
    """비트코인(KRW-BTC) 분석을 통한 시장 추세 판단"""
    try:
        btc_candles = fetch_candles("KRW-BTC", count=24)
        if len(btc_candles) < 24:
            return "NEUTRAL (보통)", 1.0

        df_btc = pd.DataFrame(btc_candles).sort_values("timestamp").reset_index(drop=True)
        btc_change = ((df_btc.iloc[-1]["trade_price"] - df_btc.iloc[0]["opening_price"]) / df_btc.iloc[0]["opening_price"]) * 100
        btc_positive = sum(1 for _, row in df_btc.iterrows() if row["trade_price"] > row["opening_price"])

        if btc_change <= -3.0 or btc_positive <= 6:
            return "BEAR (하락장 경고)", 0.85
        elif btc_change >= 2.0 and btc_positive >= 15:
            return "BULL (강세장)", 1.05
        else:
            return "NEUTRAL (보통)", 1.0
    except Exception:
        return "NEUTRAL (보통)", 1.0


def calculate_historical_win_rate(history_db, target_tp_pct=5.0, target_sl_pct=2.0):
    """과거 히스토리(TOP 10) 기반 익절/손절 승률 산출"""
    total_trades = 0
    wins = 0
    losses = 0

    now_ts = time.time()
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

            subsequent_prices = [
                r["price"] for r in records[i + 1:]
                if r["timestamp"] > entry_ts
            ]

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


def analyze_single_coin(market, k_name, ideal_pattern, history_db, weights, btc_multiplier=1.0, daily_change_rate=0.0, is_ai_recommended=False):
    ticker = market.replace("KRW-", "")

    candles = fetch_candles(market, count=672)
    if len(candles) < 60:
        return None

    df = pd.DataFrame(candles).sort_values("timestamp").reset_index(drop=True)
    df_6h = df.iloc[-24:].copy().reset_index(drop=True)
    current_price = df_6h.iloc[-1]["trade_price"]
    change_rate = daily_change_rate

    positive_count = sum(1 for _, row in df_6h.iterrows() if row["trade_price"] > row["opening_price"])

    prices = df_6h["trade_price"].values
    price_min, price_max = prices.min(), prices.max()
    norm_prices = (prices - price_min) / (price_max - price_min + 1e-8)
    distance = np.linalg.norm(norm_prices - ideal_pattern)
    pattern_similarity = max(0.0, float(1.0 - (distance / np.sqrt(len(norm_prices))))) * 100

    volume_std = df_6h["candle_acc_trade_volume"].std()
    volume_mean = df_6h["candle_acc_trade_volume"].mean()
    ai_volatility_score = float(min(100.0, (volume_std / (volume_mean + 1e-8)) * 50))

    accumulation_score = 0
    df_6h["vol_ma"] = df_6h["candle_acc_trade_volume"].rolling(window=5).mean().fillna(0)

    for i in range(1, len(df_6h)):
        row = df_6h.iloc[i]
        prev_vol_ma = df_6h.iloc[i - 1]["vol_ma"]
        if prev_vol_ma == 0:
            continue

        if row["candle_acc_trade_volume"] > prev_vol_ma * 2:
            body = abs(row["trade_price"] - row["opening_price"])
            upper_wick = row["high_price"] - max(row["trade_price"], row["opening_price"])
            lower_wick = min(row["trade_price"], row["opening_price"]) - row["low_price"]

            if lower_wick > (body * 1.5):
                accumulation_score += 30
            if row["trade_price"] > row["opening_price"] and upper_wick > (body * 2):
                accumulation_score += 20

    accumulation_score = min(100.0, accumulation_score)

    # 지표 산출 (돌파율, 거래량 급증, 이평선 정배열)
    df_24h = df.iloc[-96:] if len(df) >= 96 else df
    high_24h = df_24h["high_price"].max()
    price_to_high_ratio = current_price / (high_24h + 1e-8)
    breakout_score = 100.0 if price_to_high_ratio >= 0.98 else float(price_to_high_ratio * 100)

    df["vol_ma20"] = df["candle_acc_trade_volume"].rolling(20).mean()
    latest_vol = df.iloc[-1]["candle_acc_trade_volume"]
    prev_avg_vol = df.iloc[-2]["vol_ma20"] if len(df) >= 2 else 0.0
    vol_surge_ratio = latest_vol / (prev_avg_vol + 1e-8)
    volume_surge_score = float(min(100.0, vol_surge_ratio * 20.0))

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

    acc_24h_krw = df_24h["candle_acc_trade_price"].sum()
    liquidity_index = round(min(100.0, max(0.0, (np.log10(acc_24h_krw + 1e-8) - 7) * 20)), 1) if acc_24h_krw > 0 else 0.0

    market_history = history_db.get(market, [])
    now_ts = time.time()
    three_hours_ago = now_ts - 3 * 3600
    recent_top10_count = sum(
        1 for h in market_history if h["timestamp"] >= three_hours_ago and h["rank"] <= 10
    )

    rsi = calculate_rsi(df["trade_price"])

    overheat_penalty = 15.0 if change_rate >= 25.0 else 0.0

    base_score = (
        pattern_similarity * weights.get("w_pattern", 0.15)
        + (positive_count / 24.0 * 100) * weights.get("w_buy_sell", 0.10)
        + min(100.0, recent_top10_count * 20) * weights.get("w_recent_rank", 0.10)
        + ai_volatility_score * weights.get("w_ai_volatility", 0.05)
        + accumulation_score * weights.get("w_accumulation", 0.15)
        + breakout_score * weights.get("w_breakout", 0.20)
        + volume_surge_score * weights.get("w_vol_surge", 0.15)
        + ma_momentum_score * weights.get("w_ma_alignment", 0.10)
    )

    final_score = max(0.0, (base_score - overheat_penalty) * btc_multiplier)

    if is_ai_recommended:
        final_score *= 1.1

    return {
        "market": market,
        "ticker": ticker,
        "name": k_name,
        "current_price": current_price,
        "change_rate": round(change_rate, 2),
        "pattern_similarity": round(pattern_similarity, 1),
        "positive_count": positive_count,
        "accumulation_score": round(accumulation_score, 1),
        "score": round(final_score, 2),
        "rsi": round(rsi, 1),
        "recent_top10_count": recent_top10_count,
        "ai_volatility_score": round(ai_volatility_score, 1),
        "up_5pct_count": up_5pct_count,
        "down_5pct_count": down_5pct_count,
        "liquidity_index": liquidity_index,
        "is_ai_recommended": is_ai_recommended,
    }


def generate_upbit_r_dashboard(
    analysis_results, current_time_str, btc_status, backtest_stats, html_path="docs/index.html"
):
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    win_rate, total_trades, wins, losses = backtest_stats

    rows_list = []
    for item in analysis_results:
        change_class = (
            "plus"
            if item["change_rate"] > 0
            else ("minus" if item["change_rate"] < 0 else "")
        )
        change_sign = "+" if item["change_rate"] > 0 else ""

        ai_badge_html = (
            '<span class="ai-badge">AI추천</span>'
            if item.get("is_ai_recommended")
            else ""
        )

        rsi_display = f"{item['rsi']}"
        if item['rsi'] >= 70:
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
<td>{item['pattern_similarity']}%</td>
<td class="accumulation">{item['accumulation_score']}점</td>
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

.winrate-card {
    border-left: 6px solid #2b8a3e;
}

.winrate-val {
    font-size: 18px;
    color: #e03131;
}

.ai-badge {
    background-color: #e03131;
    color: #ffffff;
    font-size: 11px;
    font-weight: bold;
    padding: 2px 6px;
    border-radius: 4px;
    margin-left: 6px;
    display: inline-block;
    vertical-align: middle;
}

.coin-link { color: #333333; text-decoration: none; cursor: pointer; }
.coin-link:hover { color: #007bff; text-decoration: underline; }

.search-box { margin-bottom: 20px; }
.search-box input { width: 100%; padding: 12px 15px; font-size: 16px; border: 1px solid #ced4da; border-radius: 6px; box-sizing: border-box; outline: none; background: #ffffff; }

table { width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
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
.accumulation { color: #d9480f; font-weight: bold; }
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
        "interval": "15",
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
<div class="header-center"><h2 style="margin: 0; font-size: 20px;">🚀 업비트 실시간 급등주 포착 대시보드</h2></div>
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
<table id="coinTable">
<thead>
<tr>
<th onclick="sortTable(0)">순위</th>
<th onclick="sortTable(1)">한글코인명</th>
<th onclick="sortTable(2)">현재가격 (KRW)</th>
<th onclick="sortTable(3)">전일대비 등락률</th>
<th onclick="sortTable(4)">RSI(14)</th>
<th onclick="sortTable(5)">패턴유사율</th>
<th onclick="sortTable(6)">세력매집</th>
<th onclick="sortTable(7)">유동성</th>
<th onclick="sortTable(8)">최종예측점수</th>
<th onclick="sortTable(9)">5% 변동 (상승/하락)</th>
</tr>
</thead>
<tbody>
{{ROWS}}
</tbody>
</table>

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

    final_html = html_template.replace("{{CURRENT_TIME}}", current_time_str)\
                              .replace("{{BTC_STATUS}}", btc_status)\
                              .replace("{{WIN_RATE}}", str(win_rate))\
                              .replace("{{TOTAL_TRADES}}", str(total_trades))\
                              .replace("{{WINS}}", str(wins))\
                              .replace("{{LOSSES}}", str(losses))\
                              .replace("{{ROWS}}", rows_html)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"🎨 [대시보드] HTML 생성 완료 ({html_path})!")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    ai_recommend_set = fetch_ai_recommendations()

    btc_status, btc_multiplier = check_btc_market_status()
    print(f"📊 [시장 상태 진단] 비트코인 상태: {btc_status} (가중치 배율: {btc_multiplier})")

    history_db = load_json(HISTORY_FILE, {})

    backtest_stats = calculate_historical_win_rate(history_db, target_tp_pct=5.0, target_sl_pct=2.0)
    win_rate, total_trades, wins, losses = backtest_stats
    print(f"📈 [백테스팅 결과] 승률: {win_rate}% (총 {total_trades}건 / {wins}승 {losses}패)")

    weights = load_json(
        WEIGHTS_FILE,
        {
            "w_pattern": 0.15,
            "w_buy_sell": 0.10,
            "w_recent_rank": 0.10,
            "w_ai_volatility": 0.05,
            "w_accumulation": 0.15,
            "w_breakout": 0.20,
            "w_vol_surge": 0.15,
            "w_ma_alignment": 0.10,
        },
    )

    pattern_data = load_json(PATTERN_FILE, {})
    ideal_pattern = (
        np.array(pattern_data["golden_pattern"])
        if "golden_pattern" in pattern_data
        else np.linspace(0.2, 1.0, 24)
    )

    krw_markets, market_names = fetch_krw_markets()
    ticker_data = fetch_ticker_data(krw_markets)

    current_time_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    analysis_results = []

    print(f"[{current_time_str}] 멀티스레딩 분석 시작 (총 {len(krw_markets)}개 종목)...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                analyze_single_coin,
                market,
                market_names.get(market, market),
                ideal_pattern,
                history_db,
                weights,
                btc_multiplier,
                ticker_data.get(market, 0.0),
                (market.replace("KRW-", "") in ai_recommend_set or market in ai_recommend_set)
            ): market
            for market in krw_markets
        }

        for future in as_completed(futures):
            result = future.result()
            if result:
                analysis_results.append(result)

    analysis_results.sort(key=lambda x: x["score"], reverse=True)

    for idx, item in enumerate(analysis_results):
        rank = idx + 1
        item["rank"] = rank
        m_code = item["market"]
        if m_code not in history_db:
            history_db[m_code] = []
        history_db[m_code].append({
            "timestamp": time.time(),
            "score": item["score"],
            "rank": rank,
            "price": item["current_price"],
        })
        history_db[m_code] = [
            h for h in history_db[m_code] if h["timestamp"] >= time.time() - 86400
        ]

    save_json(HISTORY_FILE, history_db)
    save_json(WEIGHTS_FILE, weights)

    generate_upbit_r_dashboard(analysis_results, current_time_str, btc_status, backtest_stats, HTML_OUTPUT)
    print("대시보드 HTML 생성 완료.")


if __name__ == "__main__":
    main()
