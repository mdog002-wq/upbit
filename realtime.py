from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
import os
import time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup



# 경로 및 상수 설정
DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
DOCS_DIR = "docs"
HTML_OUTPUT = os.path.join(DOCS_DIR, "index.html")

KST = timezone(timedelta(hours=9))


def fetch_ai_recommendations():
    """GitHub 저장소의 ai_recommend_tracker.json 파싱 함수 (딕셔너리 Key 파싱 대응)"""
    url = "https://raw.githubusercontent.com/mdog002-wq/upbit-a/main/docs/ai_recommend_tracker.json"
    refined_set = set()
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            raw_tickers = []
            
            if isinstance(data, dict):
                # 1. JSON 구조가 { "BLEND": {...}, "MOCA": {...} } 형태인 경우 (최상위 Key 추출)
                # "recommended_tickers" 등의 특정 키가 존재하지 않으면 Dict의 모든 Key를 추출
                if "recommended_tickers" in data:
                    raw_tickers = data.get("recommended_tickers", [])
                elif "recommended_coins" in data:
                    raw_tickers = data.get("recommended_coins", [])
                else:
                    raw_tickers = list(data.keys())  # "BLEND", "MOCA", "W", "STX" 등 최상위 키 모두 가져옴
                    
            elif isinstance(data, list):
                raw_tickers = data

            print(f"🤖 [AI 추천 연동] 원본 수집 심볼 목록: {raw_tickers}")

            # 티커 정제 (KRW-W, W, krw-w 모두 파싱 가능하도록 저장)
            for t in raw_tickers:
                if not t or not isinstance(t, str):
                    continue
                t_str = t.strip().upper()
                refined_set.add(t_str)
                refined_set.add(t_str.replace("KRW-", ""))
                refined_set.add(f"KRW-{t_str.replace('KRW-', '')}")

            print(f"✅ [AI 추천 연동] 정제된 매핑 심볼 수: {len(refined_set)}개")
            return refined_set
        else:
            print(f"⚠️ [AI 추천 연동] HTTP 응답 코드 오류: {res.status_code}")
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


def analyze_single_coin(market, k_name, ideal_pattern, history_db, weights):
    ticker = market.replace("KRW-", "")

    candles = fetch_candles(market, count=672)
    if len(candles) < 24:
        return None

    df = pd.DataFrame(candles)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df_6h = df.iloc[-24:].copy().reset_index(drop=True)

    current_price = df_6h.iloc[-1]["trade_price"]
    prev_close = df_6h.iloc[0]["opening_price"]
    change_rate = ((current_price - prev_close) / prev_close) * 100

    positive_count = sum(
        1 for _, row in df_6h.iterrows() if row["trade_price"] > row["opening_price"]
    )

    prices = df_6h["trade_price"].values
    price_min, price_max = prices.min(), prices.max()
    norm_prices = (prices - price_min) / (price_max - price_min + 1e-8)
    distance = np.linalg.norm(norm_prices - ideal_pattern)
    pattern_similarity = (
        max(0.0, float(1.0 - (distance / np.sqrt(len(norm_prices))))) * 100
    )

    volume_std = df_6h["candle_acc_trade_volume"].std()
    volume_mean = df_6h["candle_acc_trade_volume"].mean()
    ai_volatility_score = float(
        min(100.0, (volume_std / (volume_mean + 1e-8)) * 50)
    )

    accumulation_score = 0
    df_6h["vol_ma"] = (
        df_6h["candle_acc_trade_volume"].rolling(window=5).mean().fillna(0)
    )

    for i in range(1, len(df_6h)):
        row = df_6h.iloc[i]
        prev_vol_ma = df_6h.iloc[i - 1]["vol_ma"]
        if prev_vol_ma == 0:
            continue

        if row["candle_acc_trade_volume"] > prev_vol_ma * 2:
            body = abs(row["trade_price"] - row["opening_price"])
            upper_wick = row["high_price"] - max(
                row["trade_price"], row["opening_price"]
            )
            lower_wick = (
                min(row["trade_price"], row["opening_price"]) - row["low_price"]
            )

            if lower_wick > (body * 1.5):
                accumulation_score += 30
            if (
                row["trade_price"] > row["opening_price"]
                and upper_wick > (body * 2)
            ):
                accumulation_score += 20

    accumulation_score = min(100.0, accumulation_score)

    up_5pct_count = sum(
        1
        for _, row in df.iterrows()
        if (
            (row["high_price"] - row["opening_price"])
            / (row["opening_price"] + 1e-8)
        )
        * 100
        >= 5.0
    )
    down_5pct_count = sum(
        1
        for _, row in df.iterrows()
        if (
            (row["opening_price"] - row["low_price"])
            / (row["opening_price"] + 1e-8)
        )
        * 100
        >= 5.0
    )

    df_24h = df.iloc[-96:] if len(df) >= 96 else df
    acc_24h_krw = df_24h["candle_acc_trade_price"].sum()
    if acc_24h_krw > 0:
        liquidity_index = round(
            min(100.0, max(0.0, (np.log10(acc_24h_krw) - 7) * 20)), 1
        )
    else:
        liquidity_index = 0.0

    market_history = history_db.get(market, [])
    now_ts = time.time()
    three_hours_ago = now_ts - 3 * 3600
    recent_top10_count = sum(
        1
        for h in market_history
        if h["timestamp"] >= three_hours_ago and h["rank"] <= 10
    )

    score = (
        pattern_similarity * weights["w_pattern"]
        + (positive_count / 24.0 * 100) * weights["w_buy_sell"]
        + min(100.0, recent_top10_count * 20) * weights["w_recent_rank"]
        + ai_volatility_score * weights["w_ai_volatility"]
        + accumulation_score * weights["w_accumulation"]
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
        "liquidity_index": liquidity_index,
    }

def generate_upbit_r_dashboard(
    analysis_results, current_time_str, html_path="docs/index.html"
):
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    # 1. 테이블 행(Rows) HTML 동적 생성
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

        row = f"""
<tr>
<td><b>{item['rank']}</b></td>
<td>
    <a href="#" onclick="openUpbitChart('{item['ticker']}'); return false;" class="coin-link">
        <b>{item['name']}</b> <span class="ticker-symbol">({item['ticker']})</span>
    </a>{ai_badge_html}
</td>
<td>{item['current_price']:,}</td>
<td class="{change_class}">{change_sign}{item['change_rate']}%</td>
<td>{item['pattern_similarity']}%</td>
<td class="accumulation">{item['accumulation_score']}점</td>
<td class="liquidity">{item['liquidity_index']}점</td>
<td>{item['recent_top10_count']}회</td>
<td>{item['positive_count']}회</td>
<td><span class="plus">▲{item['up_5pct_count']}회</span> / <span class="minus">▼{item['down_5pct_count']}회</span></td>
<td><b>{item['score']}점</b></td>
</tr>"""
        rows_list.append(row)

    rows_html = "".join(rows_list)

    # 2. HTML 전체 템플릿 (일반 멀티라인 문자열로 f-string 중괄호 파싱 에러 차단)
    html_template = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>업비트 실시간 급등주 포착 대시보드</title>
<style>
body { background-color: #f8f9fa; color: #333333; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; }
.header-container { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; background: #ffffff; padding: 15px 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; }
.header-left { text-align: left; } .header-center { text-align: center; } .header-right { text-align: right; font-size: 13px; color: #495057; font-weight: 500; }
.ai-btn { background-color: #007bff; color: white; padding: 10px 18px; border-radius: 5px; text-decoration: none; font-weight: bold; font-size: 14px; display: inline-block; transition: background 0.2s; }
.ai-btn:hover { background-color: #0056b3; }

/* --------- AI 추천 배지 CSS --------- */
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

/* --------- 코인명 클릭 링크 스타일 --------- */
.coin-link {
    color: #333333;
    text-decoration: none;
    cursor: pointer;
}
.coin-link:hover {
    color: #007bff;
    text-decoration: underline;
}

.search-box { margin-bottom: 20px; }
.search-box input { width: 100%; padding: 12px 15px; font-size: 16px; border: 1px solid #ced4da; border-radius: 6px; box-sizing: border-box; outline: none; background: #ffffff; }
table { width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
th, td { padding: 12px 15px; text-align: center; border-bottom: 1px solid #e9ecef; }
th { background-color: #f1f3f5; color: #495057; font-weight: 600; cursor: pointer; user-select: none; transition: background-color 0.2s; }
th:hover { background-color: #e9ecef; }
tbody tr { transition: background-color 0.15s; }
tbody tr:hover { background-color: #e9ecef !important; }
.plus { color: #e03131; font-weight: bold; }
.minus { color: #1971c2; font-weight: bold; }
.ticker-symbol { font-size: 12px; color: #868e96; font-weight: normal; margin-left: 4px; }
.accumulation { color: #d9480f; font-weight: bold; }
.liquidity { color: #2b8a3e; font-weight: bold; }
</style>
<script>
function filterTable() {
    let input = document.getElementById('searchInput').value.toLowerCase();
    let tr = document.getElementById('coinTable').getElementsByTagName('tr');
    for (let i = 1; i < tr.length; i++) {
        let td = tr[i].getElementsByTagName('td')[1];
        tr[i].style.display = (td && (td.textContent || td.innerText).toLowerCase().indexOf(input) > -1) ? "" : "none";
    }
}

// 업비트 차트 팝업 오픈 함수
function openUpbitChart(ticker) {
    const url = 'https://upbit.com/exchange?code=CRIX.UPBIT.KRW-' + ticker;
    window.open(url, 'upbitChart', 'width=1200,height=800,scrollbars=yes,resizable=yes');
}
</script>
</head>
<body>
<div class="header-container">
<div class="header-left"><a href="https://upbit-a.onrender.com" target="_self" class="ai-btn">AI리포트이동</a></div>
<div class="header-center"><h2 style="margin: 0; font-size: 20px;">🚀 업비트 실시간 급등주 포착 대시보드</h2></div>
<div class="header-right">마지막 업데이트: <b>{{CURRENT_TIME}}</b></div>
</div>
<div class="search-box"><input type="text" id="searchInput" onkeyup="filterTable()" placeholder="코인명 또는 티커 검색..."></div>
<table id="coinTable">
<thead>
<tr>
<th>순위</th><th>한글코인명</th><th>현재가격 (KRW)</th><th>전일대비등락율</th>
<th>패턴유사율</th><th>세력매집강도</th><th>유동성지수</th><th>최근 3시간 TOP10</th>
<th>매수우세</th><th>1주일 5% 변동</th><th>예측점수</th>
</tr>
</thead>
<tbody>
{{ROWS}}
</tbody>
</table>
</body>
</html>
"""

    # 3. 데이터 치환 및 파일 저장
    final_html = html_template.replace("{{CURRENT_TIME}}", current_time_str).replace("{{ROWS}}", rows_html)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"🎨 [대시보드] HTML 생성 완료 ({html_path})!")





def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # 1. upbit-a의 JSON URL에서 AI 추천 종목 가져오기
    ai_recommend_set = fetch_ai_recommendations()

    history_db = load_json(HISTORY_FILE, {})
    weights = load_json(
        WEIGHTS_FILE,
        {
            "w_pattern": 0.25,
            "w_buy_sell": 0.25,
            "w_recent_rank": 0.15,
            "w_ai_volatility": 0.15,
            "w_accumulation": 0.20,
        },
    )

    pattern_data = load_json(PATTERN_FILE, {})
    ideal_pattern = (
        np.array(pattern_data["golden_pattern"])
        if "golden_pattern" in pattern_data
        else np.linspace(0.2, 1.0, 24)
    )

    krw_markets, market_names = fetch_krw_markets()
    current_time_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    analysis_results = []

    print(
        f"[{current_time_str}] 멀티스레딩 데이터 수집 및 분석 시작 (총 {len(krw_markets)}개 종목)..."
    )

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                analyze_single_coin,
                market,
                market_names.get(market, market),
                ideal_pattern,
                history_db,
                weights,
            ): market
            for market in krw_markets
        }

        for future in as_completed(futures):
            result = future.result()
            if result:
                # 2. 결과 개체에 AI 추천 여부(is_ai_recommended) 매핑
                ticker = result["ticker"]
                market = result["market"]

                if ticker in ai_recommend_set or market in ai_recommend_set:
                    result["is_ai_recommended"] = True
                else:
                    result["is_ai_recommended"] = False

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

    generate_upbit_r_dashboard(analysis_results, current_time_str, HTML_OUTPUT)
    print("대시보드 HTML 생성 및 업데이트 완료.")


if __name__ == "__main__":
    main()
