import asyncio
import json
import websockets
import pandas as pd
import numpy as np
import requests
import time
import re
import os
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastdtw import fastdtw

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
DOCS_DIR = "docs"
HTML_OUTPUT = os.path.join(DOCS_DIR, "index.html")

KST = timezone(timedelta(hours=9))

def fetch_ai_recommendations():
    refined_set = set()
    def parse_item(item):
        if isinstance(item, str):
            match = re.search(r'\(([A-Z0-9]+)\)', item.upper())
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
    url = f"https://raw.githubusercontent.com/mdog002-wq/upbit-a/main/docs/ai_recommend_tracker.json?t={int(time.time())}"
    try:
        res = requests.get(url, timeout=5, headers={"Cache-Control": "no-cache"})
        if res.status_code == 200:
            data = res.json()
    except Exception as e:
        print(f"⚠️ 레포1 AI 추천 원격 요청 실패: {e}")

    if data:
        parse_item(data)
        
    return refined_set

def fetch_warning_coins():
    warning_set = set()
    url = f"https://raw.githubusercontent.com/mdog002-wq/upbit-a/main/docs/warning_coins.json?t={int(time.time())}"
    try:
        res = requests.get(url, timeout=5, headers={"Cache-Control": "no-cache"})
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
    except Exception as e:
        print(f"⚠️ 위험 종목 데이터 불러오기 실패: {e}")
        
    return warning_set

def calculate_dtw_similarity(seq1, seq2):
    try:
        s1 = np.asarray(seq1, dtype=np.float64).reshape(-1)
        s2 = np.asarray(seq2, dtype=np.float64).reshape(-1)
        if s1.size == 0 or s2.size == 0:
            return 0.0
        min_len = min(len(s1), len(s2))
        if min_len == 0:
            return 0.0
        s1, s2 = s1[-min_len:], s2[-min_len:]
        distance, _ = fastdtw(s1, s2, dist=lambda x, y: abs(x - y))
        avg_dist = distance / min_len
        similarity = np.exp(-1.5 * avg_dist) * 100.0
        return round(float(similarity), 1)
    except Exception:
        return 0.0

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
                                "signed_change_rate": res.get("signed_change_rate", 0) * 100
                            }
            except Exception:
                await asyncio.sleep(2)

    def start(self):
        self.is_running = False
        import threading
        self.is_running = True
        t = threading.Thread(target=lambda: asyncio.run(self._connect_websocket()), daemon=True)
        t.start()

    def stop(self):
        self.is_running = False

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

def calculate_atr(df, period=14):
    try:
        high, low, close = df['high_price'], df['low_price'], df['trade_price'].shift(1)
        tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return atr if not np.isnan(atr) else (df['trade_price'].iloc[-1] * 0.015)
    except Exception:
        return df['trade_price'].iloc[-1] * 0.015

def calculate_resistance_levels(df, current_price):
    try:
        avg_vol = df['candle_acc_trade_volume'].mean()
        high_vol_df = df[df['candle_acc_trade_volume'] > avg_vol * 1.3]
        overhead_candles = high_vol_df[high_vol_df['high_price'] > current_price]
        
        resistance_1, resistance_2 = None, None
        if not overhead_candles.empty:
            overhead_sorted = overhead_candles.sort_values(by='candle_acc_trade_volume', ascending=False)
            r1_candidates = overhead_sorted[overhead_sorted['trade_price'] > current_price]
            resistance_1 = r1_candidates.iloc[0]['trade_price'] if not r1_candidates.empty else overhead_sorted.iloc[0]['high_price']
            r2_candidates = overhead_sorted[overhead_sorted['high_price'] > resistance_1]
            if not r2_candidates.empty:
                resistance_2 = r2_candidates.iloc[0]['high_price']

        if not resistance_1: resistance_1 = df['high_price'].max()
        if not resistance_2 or resistance_2 <= resistance_1: resistance_2 = resistance_1 * 1.03
        corpse_volume_ratio = round((overhead_candles['candle_acc_trade_volume'].sum() / (df['candle_acc_trade_volume'].sum() + 1e-8)) * 100, 1)

        return resistance_1, resistance_2, corpse_volume_ratio
    except Exception:
        return current_price * 1.05, current_price * 1.10, 0.0

def check_btc_status():
    try:
        btc_candles = fetch_5m_candles("KRW-BTC", count=24)
        if len(btc_candles) < 24: return "NEUTRAL (보통)", 1.0
        df = pd.DataFrame(btc_candles).sort_values("timestamp")
        btc_change = ((df.iloc[-1]["trade_price"] - df.iloc[0]["opening_price"]) / df.iloc[0]["opening_price"]) * 100
        if btc_change <= -2.0: return "BEAR (하락장 경고)", 0.85
        elif btc_change >= 1.5: return "BULL (강세장)", 1.05
        return "NEUTRAL (보통)", 1.0
    except Exception:
        return "NEUTRAL (보통)", 1.0

def calculate_historical_win_rate(history_db, target_tp_pct=3.0, target_sl_pct=2.5):
    total_trades, wins, losses = 0, 0, 0
    for market, records in history_db.items():
        if len(records) < 2: continue
        for i in range(len(records) - 1):
            entry = records[i]
            if entry.get("rank", 99) > 10: continue
            entry_price, entry_ts = entry.get("price"), entry.get("timestamp")
            if not entry_price or entry_price <= 0: continue
            subsequent_prices = [r["price"] for r in records[i+1:] if r["timestamp"] > entry_ts]
            if not subsequent_prices: continue
            
            max_return = ((max(subsequent_prices) - entry_price) / entry_price) * 100
            min_return = ((min(subsequent_prices) - entry_price) / entry_price) * 100

            if max_return >= target_tp_pct:
                wins += 1; total_trades += 1
            elif min_return <= -target_sl_pct:
                losses += 1; total_trades += 1

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    return round(win_rate, 1), total_trades, wins, losses

def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return default

def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def analyze_single_coin(market, k_name, ideal_price_pattern, ideal_vol_pattern, history_db, weights, btc_multiplier, ws_data, is_ai_recommended, is_warning):
    ticker = market.replace("KRW-", "")
    candles = fetch_5m_candles(market, count=120)
    if len(candles) < 60: return None

    df = pd.DataFrame(candles).sort_values("timestamp").reset_index(drop=True)
    if ws_data and "trade_price" in ws_data:
        current_price, change_rate = ws_data["trade_price"], ws_data["signed_change_rate"]
    else:
        current_price = df.iloc[-1]["trade_price"]
        change_rate = ((current_price - df.iloc[-1]["prev_closing_price"]) / df.iloc[-1]["prev_closing_price"]) * 100

    df_frame = df.iloc[-36:].copy().reset_index(drop=True)
    prices, volumes = df_frame["trade_price"].values, df_frame["candle_acc_trade_volume"].values

    p_range = (prices.max() - prices.min()) if (prices.max() - prices.min()) > 1e-8 else 1.0
    v_range = (volumes.max() - volumes.min()) if (volumes.max() - volumes.min()) > 1e-8 else 1.0

    norm_prices = (prices - prices.min()) / p_range
    norm_volumes = (volumes - volumes.min()) / v_range

    price_sim = calculate_dtw_similarity(norm_prices, ideal_price_pattern)
    vol_sim = calculate_dtw_similarity(norm_volumes, ideal_vol_pattern)
    combined_pattern_sim = round(price_sim * 0.7 + vol_sim * 0.3, 1)

    positive_count = sum(1 for _, row in df_frame.iterrows() if row["trade_price"] > row["opening_price"])
    ai_volatility_score = float(min(100.0, (df_frame["candle_acc_trade_volume"].std() / (df_frame["candle_acc_trade_volume"].mean() + 1e-8)) * 50))

    recent_vol_current = df.iloc[-1]["candle_acc_trade_volume"]
    avg_prev_vol = df.iloc[-21:-1]["candle_acc_trade_volume"].mean()
    vol_cliff_score = min(100.0, max(0.0, (1.0 - (recent_vol_current / avg_prev_vol)) * 100.0)) if avg_prev_vol > 0 else 0.0

    high_24h = df["high_price"].max()
    breakout_score = 100.0 if current_price >= high_24h else (current_price / high_24h) * 100

    recent_vol = df.iloc[-3:]["candle_acc_trade_volume"].sum()
    avg_vol = df.iloc[-36:-3]["candle_acc_trade_volume"].sum() / 33 * 3
    vol_surge_score = min(100.0, (recent_vol / (avg_vol + 1e-8)) * 25.0)

    df["ma5"] = df["trade_price"].rolling(5).mean()
    df["ma20"] = df["trade_price"].rolling(20).mean()
    df["ma60"] = df["trade_price"].rolling(60).mean()
    last_row = df.iloc[-1]

    dev_5_20 = ((last_row["ma5"] - last_row["ma20"]) / (last_row["ma20"] + 1e-8)) * 100
    ma_momentum_score = min(100.0, max(0.0, 50.0 + (dev_5_20 * 20.0)))

    up_5pct_count = sum(1 for _, row in df.iterrows() if ((row["high_price"] - row["low_price"]) / (row["low_price"] + 1e-8)) * 100 >= 5.0 and row["trade_price"] >= row["opening_price"])
    down_5pct_count = sum(1 for _, row in df.iterrows() if ((row["high_price"] - row["low_price"]) / (row["high_price"] + 1e-8)) * 100 >= 5.0 and row["trade_price"] < row["opening_price"])

    acc_24h_krw = df["candle_acc_trade_price"].sum()
    liquidity_index = round(min(100.0, max(0.0, (np.log10(acc_24h_krw + 1e-8) - 7) * 20)), 1) if acc_24h_krw > 0 else 0.0
    rsi = calculate_rsi(df["trade_price"])

    raw_score = (
        combined_pattern_sim * weights.get("w_pattern", 0.25) +
        (positive_count / 36.0 * 100) * weights.get("w_buy_sell", 0.10) +
        ai_volatility_score * weights.get("w_ai_volatility", 0.05) +
        vol_cliff_score * weights.get("w_vol_cliff", 0.15) +
        breakout_score * weights.get("w_breakout", 0.05) +
        vol_surge_score * weights.get("w_vol_surge", 0.10) +
        ma_momentum_score * weights.get("w_ma_alignment", 0.15) +
        min(100.0, max(0.0, change_rate * 3.33)) * weights.get("w_daily_momentum", 0.15)
    )

    if rsi >= 68.0: raw_score *= 0.60
    elif rsi <= 35.0: raw_score *= 0.80
    
    if change_rate >= 12.0: raw_score *= 0.50
    if liquidity_index < 15.0: raw_score *= 0.50

    # -------------------------------------------------------------
    # 최근 3시간 (5분 간격 = 최근 36회 실행) 동안 TOP 10 진입 횟수 계산
    # -------------------------------------------------------------
    prev_history = history_db.get(market, [])
    prev_score = prev_history[-1]["score"] if prev_history else None

    recent_36_records = prev_history[-36:] if prev_history else []
    top10_count_3h = sum(1 for r in recent_36_records if r.get("rank", 99) <= 10)

    if prev_score is not None:
        smoothed_score = (raw_score * 0.4) + (prev_score * 0.6)
    else:
        smoothed_score = raw_score

    # 최근 3시간 중 TOP 10에 머문 비율에 따라 최대 +5.0점 가산점 부여
    count_bonus = min(5.0, (top10_count_3h / 36.0) * 5.0)
    smoothed_score += count_bonus

    final_score = max(0.0, smoothed_score * btc_multiplier)
    if is_ai_recommended:
        final_score *= 1.01 if (rsi < 68.0 and vol_surge_score >= 10.0) else 0.95

    atr = calculate_atr(df, period=14)
    res_1, res_2, corpse_ratio = calculate_resistance_levels(df, current_price)
    if corpse_ratio >= 40.0: final_score *= 0.90

    final_score = round(final_score, 2)
    
    calculated_tp2 = max(current_price * 1.03, min(res_2 * 0.998, current_price + (atr * 3.0)))
    calculated_max_tp = max(calculated_tp2 * 1.03, max(df["high_price"].max(), current_price + (atr * 5.0)))
    calculated_sl = min(current_price * 0.975, min(current_price - (atr * 1.2), df_frame["low_price"].min() * 0.995))

    return {
        "market": market, "ticker": ticker, "name": k_name,
        "current_price": current_price, "change_rate": round(change_rate, 2),
        "pattern_similarity": combined_pattern_sim, "positive_count": positive_count,
        "vol_cliff_score": round(vol_cliff_score, 1), "score": final_score,
        "rsi": round(rsi, 1), "ai_volatility_score": round(ai_volatility_score, 1),
        "up_5pct_count": up_5pct_count, "down_5pct_count": down_5pct_count,
        "liquidity_index": liquidity_index, "is_ai_recommended": is_ai_recommended,
        "is_warning": is_warning, "top10_count": top10_count_3h,
        "tp2": round(calculated_tp2, 4), "max_tp": round(calculated_max_tp, 4),
        "sl": round(calculated_sl, 4),
        "tp2_pct": round(((calculated_tp2 - current_price) / current_price) * 100, 2),
        "max_tp_pct": round(((calculated_max_tp - current_price) / current_price) * 100, 2),
        "sl_pct": round(((calculated_sl - current_price) / current_price) * 100, 2),
        "corpse_ratio": corpse_ratio, "res_1": round(res_1, 4), "res_2": round(res_2, 4)
    }

def generate_full_dashboard_html(analysis_results, current_time_str, btc_status, backtest_stats, html_path=HTML_OUTPUT):
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    win_rate, total_trades, wins, losses = backtest_stats
    dashboard_json_data = json.dumps(analysis_results, ensure_ascii=False)

    rows_list = []
    for item in analysis_results:
        change_class = "plus" if item["change_rate"] > 0 else ("minus" if item["change_rate"] < 0 else "")
        change_sign = "+" if item["change_rate"] > 0 else ""
        ai_badge_html = '<span class="ai-badge">AI추천</span>' if item.get("is_ai_recommended") else ""
        warning_badge_html = '<span class="warning-badge">⚠️ 위험</span>' if item.get("is_warning") else ""
        rsi_display = f"{item['rsi']}" if item["rsi"] < 68 else f"<span class='overheat'>{item['rsi']} (과열)</span>"

        row = f"""
<tr>
<td><b>{item['rank']}</b></td>
<td>
    <a href="#" onclick="openChartModal('{item['ticker']}', '{item['name']}'); return false;" class="coin-link">
        <b>{item['name']}</b> <span class="ticker-symbol">({item['ticker']})</span>
    </a>{ai_badge_html}{warning_badge_html}
</td>
<td>{item['current_price']:,}</td>
<td class="{change_class}">{change_sign}{item['change_rate']}%</td>
<td>{rsi_display}</td>
<td><b>{item['pattern_similarity']}%</b></td>
<td class="vol-cliff">{item['vol_cliff_score']}점</td>
<td class="top10-count"><b>{item['top10_count']}회</b> <span style="font-size:11px; color:#888;">/3시간</span></td>
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
  <meta http-equiv="refresh" content="240">
<style>
body { background-color: #f8f9fa; color: #333333; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; }
.header-container { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; background: #ffffff; padding: 15px 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 15px; }
.header-left { text-align: left; } .header-center { text-align: center; } .header-right { text-align: right; font-size: 13px; color: #495057; font-weight: 500; }
.ai-btn { background-color: #007bff; color: white; padding: 10px 18px; border-radius: 5px; text-decoration: none; font-weight: bold; font-size: 14px; display: inline-block; transition: background 0.2s; }
.ai-btn:hover { background-color: #0056b3; }

.status-card { background: #ffffff; padding: 12px 20px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); font-size: 14px; font-weight: bold; color: #495057; }
.winrate-card { border-left: 6px solid #2b8a3e; }
.winrate-val { font-size: 18px; color: #e03131; }

.ai-badge { background-color: #0d6efd !important; color: #ffffff !important; font-size: 11px !important; font-weight: bold !important; padding: 2px 6px !important; border-radius: 4px !important; margin-left: 6px !important; display: inline-block !important; vertical-align: middle !important; }
.warning-badge { background-color: #ff4d4f !important; color: #ffffff !important; font-size: 11px !important; font-weight: bold !important; padding: 2px 6px !important; border-radius: 4px !important; margin-left: 4px !important; display: inline-block !important; vertical-align: middle !important; animation: pulse 1.5s infinite; }

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.6; }
  100% { opacity: 1; }
}

.coin-link { color: #333333; text-decoration: none; cursor: pointer; }
.coin-link:hover { color: #007bff; text-decoration: underline; }

.toolbar-container { display: flex; justify-content: space-between; align-items: center; gap: 15px; margin-bottom: 15px; }
.search-box { flex: 1; }
.search-box input { width: 100%; padding: 10px 15px; font-size: 15px; border: 1px solid #ced4da; border-radius: 6px; outline: none; background: #ffffff; box-sizing: border-box; }

.ai-diagnosis-box { display: flex; align-items: center; gap: 8px; background: #ffffff; padding: 6px 12px; border-radius: 6px; border: 1px solid #ced4da; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.ai-diagnosis-box input { padding: 8px 10px; border: 1px solid #ced4da; border-radius: 4px; font-size: 14px; outline: none; }
.ai-diagnosis-btn { background-color: #2b8a3e; color: white; border: none; padding: 8px 14px; border-radius: 4px; font-weight: bold; cursor: pointer; font-size: 14px; transition: background 0.2s; }
.ai-diagnosis-btn:hover { background-color: #216a2f; }

.ai-result-card { display: none; background: #eef3fc; border: 1px solid #b3d4ff; padding: 12px 20px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }

.table-container { max-height: 75vh; overflow-y: auto; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); background: #ffffff; }
table { width: 100%; border-collapse: collapse; background: #ffffff; }
th, td { padding: 12px 15px; text-align: center; border-bottom: 1px solid #e9ecef; }
th { position: sticky; top: 0; z-index: 10; background-color: #f1f3f5; color: #495057; font-weight: 600; cursor: pointer; user-select: none; transition: background-color 0.2s; box-shadow: inset 0 -1px 0 #e9ecef; }
th:hover { background-color: #e9ecef; }
tbody tr { transition: background-color 0.15s; }
tbody tr:hover { background-color: #e9ecef !important; }
.plus { color: #e03131; font-weight: bold; }
.minus { color: #1971c2; font-weight: bold; }
.overheat { color: #d9480f; font-weight: bold; }
.ticker-symbol { font-size: 12px; color: #868e96; font-weight: normal; margin-left: 4px; }
.vol-cliff { color: #d9480f; font-weight: bold; }
.liquidity { color: #2b8a3e; font-weight: bold; }
.top10-count { color: #0d6efd; font-weight: bold; }

.modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.6); z-index: 9999; justify-content: center; align-items: center; }
.modal-content { background: #ffffff; width: 90%; max-width: 1000px; height: 650px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.3); display: flex; flex-direction: column; overflow: hidden; }
.modal-header { padding: 15px 20px; background: #1e222d; color: #ffffff; display: flex; justify-content: space-between; align-items: center; }
.modal-title { font-size: 18px; font-weight: bold; }
.modal-close { font-size: 24px; cursor: pointer; color: #cccccc; line-height: 1; }
.modal-close:hover { color: #ffffff; }
.modal-body { flex: 1; width: 100%; height: 100%; background: #131722; }
</style>
<script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
<script>
const dashboardData = {{DASHBOARD_JSON_DATA}};

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
            return isAscending ? cellA.localeCompare(cellB, 'ko-KR') : cellB.localeCompare(cellA, 'ko-KR');
        }
    });

    rows.forEach(row => tbody.appendChild(row));
}

function runAiDiagnosis() {
    const inputKeyword = document.getElementById('aiCoinInput').value.trim().toUpperCase();
    const entryPrice = parseFloat(document.getElementById('aiPriceInput').value);
    const resultCard = document.getElementById('aiResultCard');

    if (!inputKeyword || isNaN(entryPrice) || entryPrice <= 0) {
        alert('올바른 종목명/티커와 진입가를 입력해주세요.');
        return;
    }

    const coin = dashboardData.find(item => 
        item.ticker.toUpperCase() === inputKeyword || 
        item.name.toUpperCase() === inputKeyword || 
        item.market.toUpperCase() === 'KRW-' + inputKeyword
    );

    if (!coin) {
        alert('대시보드 리스트에서 해당 코인을 찾을 수 없습니다.');
        return;
    }

    const ratio = entryPrice / coin.current_price;
    const tp2 = coin.tp2 * ratio;
    const maxTp = coin.max_tp * ratio;
    const sl = coin.sl * ratio;

    let comment = `📊 예측 점수 <strong>${coin.score}점</strong>, DTW 유사도 <strong>${coin.pattern_similarity}%</strong>차트입니다. (최근 3시간 내 TOP10 진입: <strong>${coin.top10_count}회</strong>)<br>`;
    if (coin.is_warning) {
        comment += `🚨 <strong>위험 코인 경고:</strong> 이 코인은 위험 코인 목록(warning_coins.json)에 지정되어 있으므로 변동성에 각별히 주의하세요!<br>`;
    }
    if (coin.corpse_ratio >= 35.0) {
        comment += `⚠️ <strong>상방 저항:</strong> 물린 매물대(시체 비중 ${coin.corpse_ratio}%)로 인해 1차 저항선(${coin.res_1.toLocaleString()} KRW) 돌파가 중요합니다.`;
    } else {
        comment += `✅ <strong>매물대 양호:</strong> 저항이 가벼워(시체 비중 ${coin.corpse_ratio}%) 지표상 최대 목표가(Max TP)까지 상승 시도가 가능합니다.`;
    }

    resultCard.style.display = 'block';
    resultCard.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <strong style="font-size: 16px; color: #1e222d;">🤖 [${coin.name} / ${coin.ticker}] AI 스마트 진단 리포트</strong>
                <span style="font-size: 13px; color: #555; margin-left: 8px;">(현재가: ${coin.current_price.toLocaleString()} KRW)</span>
            </div>
            <div style="font-size: 13px; display: flex; gap: 10px; flex-wrap: wrap;">
                <span style="color: #2b8a3e; font-weight: bold;">🎯 목표가: ${tp2.toLocaleString(undefined, {maximumFractionDigits: 2})} (+${coin.tp2_pct}%)</span>
                <span style="color: #007bff; font-weight: bold;">🚀 Max TP: ${maxTp.toLocaleString(undefined, {maximumFractionDigits: 2})} (+${coin.max_tp_pct}%)</span>
                <span style="color: #e03131; font-weight: bold;">🛑 손절가: ${sl.toLocaleString(undefined, {maximumFractionDigits: 2})} (${coin.sl_pct}%)</span>
            </div>
        </div>
        <div style="margin-top: 8px; font-size: 13px; color: #333; background: #ffffff; padding: 8px 12px; border-radius: 4px; border-left: 4px solid #007bff;">
            ${comment}
        </div>
    `;
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
<div class="header-left"><a href="https://mdog002-wq.github.io/upbit-a/" target="_self" class="ai-btn">AI리포트이동</a></div>
<div class="header-center"><h2 style="margin: 0; font-size: 20px;">🚀 실시간 DTW + 웹소켓 고도화 대시보드</h2></div>
<div class="header-right">마지막 업데이트: <b>{{CURRENT_TIME}}</b></div>
</div>

<div class="status-card winrate-card">
🎯 <b>실시간 백테스팅 승률 (익절 +3% / 손절 -2.5% 기준):</b> 
<span class="winrate-val">{{WIN_RATE}}%</span> 
<span style="font-size: 13px; color: #666; font-weight: normal;">(최근 포착 TOP10 종목 총 {{TOTAL_TRADES}}건 검증 — {{WINS}}승 {{LOSSES}}패)</span>
</div>

<div class="status-card">
🌐 비트코인(BTC) 시장 상황: <span style="color:#007bff;">{{BTC_STATUS}}</span>
</div>

<div class="toolbar-container">
    <div class="search-box">
        <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="코인명 또는 티커 검색...">
    </div>
    <div class="ai-diagnosis-box">
        <span style="font-weight: bold; font-size: 13px; color: #007bff;">🤖 AI 스마트 진단:</span>
        <input type="text" id="aiCoinInput" placeholder="종목명/티커" style="width: 100px;">
        <input type="number" id="aiPriceInput" placeholder="진입가 (KRW)" style="width: 110px;">
        <button class="ai-diagnosis-btn" onclick="runAiDiagnosis()">분석하기</button>
    </div>
</div>

<div id="aiResultCard" class="ai-result-card"></div>

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
<th onclick="sortTable(7)">10위내 횟수</th>
<th onclick="sortTable(8)">유동성</th>
<th onclick="sortTable(9)">최종예측점수</th>
<th onclick="sortTable(10)">5% 변동 (상승/하락)</th>
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

    final_html = html_template.replace("{{CURRENT_TIME}}", current_time_str)\
                              .replace("{{BTC_STATUS}}", btc_status)\
                              .replace("{{WIN_RATE}}", str(win_rate))\
                              .replace("{{TOTAL_TRADES}}", str(total_trades))\
                              .replace("{{WINS}}", str(wins))\
                              .replace("{{LOSSES}}", str(losses))\
                              .replace("{{DASHBOARD_JSON_DATA}}", dashboard_json_data)\
                              .replace("{{ROWS}}", rows_html)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    krw_markets, market_names = fetch_krw_markets()
    ws_manager = UpbitWebSocketManager(krw_markets)
    ws_manager.start()
    time.sleep(2)

    ai_recommend_set = fetch_ai_recommendations()
    warning_coin_set = fetch_warning_coins()
    btc_status, btc_multiplier = check_btc_status()

    history_db = load_json(HISTORY_FILE, {})
    backtest_stats = calculate_historical_win_rate(history_db, target_tp_pct=3.0, target_sl_pct=2.5)

    weights = load_json(WEIGHTS_FILE, {
        "w_pattern": 0.25,
        "w_buy_sell": 0.10, 
        "w_ai_volatility": 0.05,
        "w_vol_cliff": 0.15,
        "w_breakout": 0.05,
        "w_vol_surge": 0.10,
        "w_ma_alignment": 0.15, 
        "w_daily_momentum": 0.15
    })

    pattern_data = load_json(PATTERN_FILE, {})
    raw_price = pattern_data.get("golden_pattern", np.linspace(0.2, 1.0, 36).tolist())
    raw_vol = pattern_data.get("golden_volume_pattern", np.linspace(0.1, 1.0, 36).tolist())

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
                (market in ai_recommend_set or market.replace("KRW-", "") in ai_recommend_set),
                (market in warning_coin_set or market.replace("KRW-", "") in warning_coin_set)
            ): market for market in krw_markets
        }
        for future in as_completed(futures):
            res = future.result()
            if res: analysis_results.append(res)

    ws_manager.stop()
    analysis_results.sort(key=lambda x: x["score"], reverse=True)

    for idx, item in enumerate(analysis_results):
        rank = idx + 1
        item["rank"] = rank
        m_code = item["market"]
        if m_code not in history_db: history_db[m_code] = []
        history_db[m_code].append({
            "timestamp": time.time(), "score": item["score"], "rank": rank, "price": item["current_price"]
        })
        # 24시간 분량(약 288회)의 히스토리를 보관하여 3시간 분량(-36) 계산에 부족함이 없도록 유지
        history_db[m_code] = [h for h in history_db[m_code] if h["timestamp"] >= time.time() - 86400][-300:]

    save_json(HISTORY_FILE, history_db)
    save_json(WEIGHTS_FILE, weights)

    current_time_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    generate_full_dashboard_html(analysis_results, current_time_str, btc_status, backtest_stats, HTML_OUTPUT)
    print("🎨 [대시보드 업데이트 완료] 기준 시간 3시간으로 변경 완료!")

if __name__ == "__main__":
    main()
