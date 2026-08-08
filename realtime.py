import os
import json
import time
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Upbit AI Quantitative Dashboard Server")

# 경로 및 상수 설정
DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
DOCS_DIR = "docs"
HTML_OUTPUT = os.path.join(DOCS_DIR, "index.html")

KST = timezone(timedelta(hours=9))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


# ==============================================================================
# [헬퍼 및 데이터 수집 함수]
# ==============================================================================
def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return default

def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def fetch_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    res = requests.get(url)
    if res.status_code != 200:
        return [], {}
    markets = res.json()
    krw_markets = [m['market'] for m in markets if m['market'].startswith("KRW-")]
    market_names = {m['market']: m['korean_name'] for m in markets if m['market'].startswith("KRW-")}
    return krw_markets, market_names

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
            to_param = data[-1]['candle_date_time_utc']
        except Exception:
            time.sleep(0.2)
            continue
            
    return all_candles

def analyze_single_coin(market, k_name, ideal_pattern, history_db, weights):
    ticker = market.replace("KRW-", "")
    
    candles = fetch_candles(market, count=672)
    if len(candles) < 24:
        return None

    df = pd.DataFrame(candles)
    df = df.sort_values('timestamp').reset_index(drop=True)

    df_6h = df.iloc[-24:].copy().reset_index(drop=True)

    current_price = df_6h.iloc[-1]['trade_price']
    prev_close = df_6h.iloc[0]['opening_price']
    change_rate = ((current_price - prev_close) / prev_close) * 100

    positive_count = sum(1 for _, row in df_6h.iterrows() if row['trade_price'] > row['opening_price'])

    prices = df_6h['trade_price'].values
    price_min, price_max = prices.min(), prices.max()
    norm_prices = (prices - price_min) / (price_max - price_min + 1e-8)
    distance = np.linalg.norm(norm_prices - ideal_pattern)
    pattern_similarity = max(0.0, float(1.0 - (distance / np.sqrt(len(norm_prices))))) * 100

    volume_std = df_6h['candle_acc_trade_volume'].std()
    volume_mean = df_6h['candle_acc_trade_volume'].mean()
    ai_volatility_score = float(min(100.0, (volume_std / (volume_mean + 1e-8)) * 50))

    accumulation_score = 0
    df_6h['vol_ma'] = df_6h['candle_acc_trade_volume'].rolling(window=5).mean().fillna(0)
    
    for i in range(1, len(df_6h)):
        row = df_6h.iloc[i]
        prev_vol_ma = df_6h.iloc[i-1]['vol_ma']
        if prev_vol_ma == 0: continue
        
        if row['candle_acc_trade_volume'] > prev_vol_ma * 2:
            body = abs(row['trade_price'] - row['opening_price'])
            upper_wick = row['high_price'] - max(row['trade_price'], row['opening_price'])
            lower_wick = min(row['trade_price'], row['opening_price']) - row['low_price']
            
            if lower_wick > (body * 1.5):
                accumulation_score += 30
            if row['trade_price'] > row['opening_price'] and upper_wick > (body * 2):
                accumulation_score += 20

    accumulation_score = min(100.0, accumulation_score)

    up_5pct_count = sum(1 for _, row in df.iterrows() if ((row['high_price'] - row['opening_price']) / (row['opening_price'] + 1e-8)) * 100 >= 5.0)
    down_5pct_count = sum(1 for _, row in df.iterrows() if ((row['opening_price'] - row['low_price']) / (row['opening_price'] + 1e-8)) * 100 >= 5.0)

    df_24h = df.iloc[-96:] if len(df) >= 96 else df
    acc_24h_krw = df_24h['candle_acc_trade_price'].sum()
    if acc_24h_krw > 0:
        liquidity_index = round(min(100.0, max(0.0, (np.log10(acc_24h_krw) - 7) * 20)), 1)
    else:
        liquidity_index = 0.0

    market_history = history_db.get(market, [])
    now_ts = time.time()
    three_hours_ago = now_ts - 3 * 3600
    recent_top10_count = sum(1 for h in market_history if h['timestamp'] >= three_hours_ago and h['rank'] <= 10)

    score = (
        pattern_similarity * weights["w_pattern"] +
        (positive_count / 24.0 * 100) * weights["w_buy_sell"] +
        min(100.0, recent_top10_count * 20) * weights["w_recent_rank"] +
        ai_volatility_score * weights["w_ai_volatility"] +
        accumulation_score * weights["w_accumulation"]
    )

    return {
        "market": market,
        "ticker": ticker,
        "name": k_name,
        "current_price": current_price,
        "change_rate": round(change_rate, 2),
        "pattern_similarity": round(pattern_similarity, 1),
        "positive_count": positive_count,
        "accumulation_score": round(accumulation_score, 1),
        "score": round(score, 2),
        "recent_top10_count": recent_top10_count,
        "ai_volatility_score": round(ai_volatility_score, 1),
        "up_5pct_count": up_5pct_count,
        "down_5pct_count": down_5pct_count,
        "liquidity_index": liquidity_index
    }


# ==============================================================================
# [FastAPI 라우터 설정]
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """메인 대시보드 페이지 반환"""
    if os.path.exists(HTML_OUTPUT):
        with open(HTML_OUTPUT, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>대시보드가 아직 생성되지 않았습니다. 잠시 후 다시 시도해 주세요.</h3>"


@app.get("/detail", response_class=HTMLResponse)
async def get_1st_site_detail_view(request: Request, symbol: str = "KRW-BTC"):
    """
    모달 내 iframe 전용 종목 상세 분석 라우터
    실제 분석 함수를 즉시 실행하여 실시간 데이터를 랜더링합니다.
    """
    symbol_upper = symbol.upper()
    market = symbol_upper if symbol_upper.startswith("KRW-") else f"KRW-{symbol_upper}"
    ticker = market.replace("KRW-", "")

    history_db = load_json(HISTORY_FILE, {})
    weights = load_json(WEIGHTS_FILE, {
        "w_pattern": 0.25, "w_buy_sell": 0.25, "w_recent_rank": 0.15,
        "w_ai_volatility": 0.15, "w_accumulation": 0.20
    })
    pattern_data = load_json(PATTERN_FILE, {})
    ideal_pattern = np.array(pattern_data["golden_pattern"]) if "golden_pattern" in pattern_data else np.linspace(0.2, 1.0, 24)

    # 해당 코인 단일 실시간 분석 실행
    analyzed_data = analyze_single_coin(market, ticker, ideal_pattern, history_db, weights)

    if analyzed_data:
        coin_info = {
            "symbol": ticker,
            "price": f"{analyzed_data['current_price']:,}",
            "score": analyzed_data['score'],
            "pattern_match": f"{analyzed_data['pattern_similarity']}%",
            "volume_power": f"{analyzed_data['accumulation_score']} 점"
        }
    else:
        coin_info = {
            "symbol": ticker,
            "price": "N/A",
            "score": 0.0,
            "pattern_match": "0.0%",
            "volume_power": "0 점"
        }

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #ffffff; padding: 15px; font-family: sans-serif; }}
            .card {{ border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            .metric-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f1f5f9; }}
        </style>
    </head>
    <body>
        <div class="card p-3">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h6 class="fw-bold m-0 text-primary">📊 1번 사이트 지표 요약</h6>
                <span class="badge bg-primary">{coin_info['symbol']}</span>
            </div>
            <hr class="my-2">
            
            <div class="metric-row">
                <span class="text-secondary">현재 가격</span>
                <span class="fw-bold">{coin_info['price']} 원</span>
            </div>
            <div class="metric-row">
                <span class="text-secondary">예측 점수</span>
                <span class="fw-bold text-primary">{coin_info['score']} 점</span>
            </div>
            <div class="metric-row">
                <span class="text-secondary">패턴 유사율</span>
                <span class="fw-bold">{coin_info['pattern_match']}</span>
            </div>
            <div class="metric-row">
                <span class="text-secondary">세력 매집 강도</span>
                <span class="fw-bold text-success">{coin_info['volume_power']}</span>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# ==============================================================================
# [실행 진입점]
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
