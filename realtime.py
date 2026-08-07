import os
import json
import time
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import pandas as pd

# 경로 및 상수 설정
DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
DOCS_DIR = "docs"
HTML_OUTPUT = os.path.join(DOCS_DIR, "index.html")

KST = timezone(timedelta(hours=9))

def fetch_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    res = requests.get(url)
    if res.status_code != 200:
        return [], {}
    markets = res.json()
    krw_markets = [m['market'] for m in markets if m['market'].startswith("KRW-")]
    market_names = {m['market']: m['korean_name'] for m in markets if m['market'].startswith("KRW-")}
    return krw_markets, market_names

def fetch_candles(market, count=24):
    url = f"https://api.upbit.com/v1/candles/minutes/15?market={market}&count={count}"
    res = requests.get(url)
    if res.status_code != 200:
        return []
    return res.json()

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

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # 1. 초기 데이터 및 가중치 불러오기 (최초 실행 시 오류 방지)
    history_db = load_json(HISTORY_FILE, {}) 
    weights = load_json(WEIGHTS_FILE, {
        "w_pattern": 0.25,
        "w_buy_sell": 0.25,
        "w_recent_rank": 0.15,
        "w_ai_volatility": 0.15,
        "w_accumulation": 0.20
    })

    # 학습된 과거 6개월 급등 황금 패턴 불러오기 (없으면 기본 우상향 패턴 적용)
    pattern_data = load_json(PATTERN_FILE, {})
    if "golden_pattern" in pattern_data:
        ideal_pattern = np.array(pattern_data["golden_pattern"])
    else:
        ideal_pattern = np.linspace(0.2, 1.0, 24)

    krw_markets, market_names = fetch_krw_markets()
    current_time_str = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')
    analysis_results = []

    print(f"[{current_time_str}] 데이터 수집 및 분석 시작 (총 {len(krw_markets)}개 종목)...")

    for market in krw_markets:
        k_name = market_names.get(market, market)
        candles = fetch_candles(market, count=24) # 최근 6시간 15분봉
        
        if len(candles) < 24:
            continue

        df = pd.DataFrame(candles)
        df = df.sort_values('timestamp').reset_index(drop=True)

        current_price = df.iloc[-1]['trade_price']
        prev_close = df.iloc[0]['opening_price']
        change_rate = ((current_price - prev_close) / prev_close) * 100

        # 최근 6시간 매수 우세 횟수 계산 (양봉 캔들 횟수)
        positive_count = sum(1 for _, row in df.iterrows() if row['trade_price'] > row['opening_price'])

        # 학습된 급등 패턴과의 유사도 계산
        prices = df['trade_price'].values
        price_min, price_max = prices.min(), prices.max()
        norm_prices = (prices - price_min) / (price_max - price_min + 1e-8)
        distance = np.linalg.norm(norm_prices - ideal_pattern)
        pattern_similarity = max(0.0, float(1.0 - (distance / np.sqrt(len(norm_prices))))) * 100

        # AI 추천용 거래량 변동성 점수
        volume_std = df['candle_acc_trade_volume'].std()
        volume_mean = df['candle_acc_trade_volume'].mean()
        ai_volatility_score = float(min(100.0, (volume_std / (volume_mean + 1e-8)) * 50))

        # 세력 매집 추적 점수 (거래량 폭발 및 꼬리 패턴)
        accumulation_score = 0
        df['vol_ma'] = df['candle_acc_trade_volume'].rolling(window=5).mean().fillna(0)
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_vol_ma = df.iloc[i-1]['vol_ma']
            if prev_vol_ma == 0: continue
            
            if row['candle_acc_trade_volume'] > prev_vol_ma * 2:
                body = abs(row['trade_price'] - row['opening_price'])
                upper_wick = row['high_price'] - max(row['trade_price'], row['opening_price'])
                lower_wick = min(row['trade_price'], row['opening_price']) - row['low_price']
                
                # 밑꼬리 수급 흡수
                if lower_wick > (body * 1.5):
                    accumulation_score += 30
                # 매물대 테스트 매집봉
                if row['trade_price'] > row['opening_price'] and upper_wick > (body * 2):
                    accumulation_score += 20

        accumulation_score = min(100.0, accumulation_score)

        # 최근 3시간 이내 TOP 10 포함 횟수 계산
        market_history = history_db.get(market, [])
        now_ts = time.time()
        three_hours_ago = now_ts - 3 * 3600
        recent_top10_count = sum(1 for h in market_history if h['timestamp'] >= three_hours_ago and h['rank'] <= 10)

        # 가중치 기반 최종 예측 점수 산출
        score = (
            pattern_similarity * weights["w_pattern"] +
            (positive_count / 24.0 * 100) * weights["w_buy_sell"] +
            min(100.0, recent_top10_count * 20) * weights["w_recent_rank"] +
            ai_volatility_score * weights["w_ai_volatility"] +
            accumulation_score * weights["w_accumulation"]
        )

        analysis_results.append({
            "market": market,
            "name": k_name,
            "current_price": current_price,
            "change_rate": round(change_rate, 2),
            "pattern_similarity": round(pattern_similarity, 1),
            "positive_count": positive_count,
            "accumulation_score": round(accumulation_score, 1),
            "score": round(score, 2),
            "recent_top10_count": recent_top10_count,
            "ai_volatility_score": round(ai_volatility_score, 1)
        })
        
        time.sleep(0.04)

    # 예측 점수 순 정렬 및 순위 부여
    analysis_results.sort(key=lambda x: x['score'], reverse=True)

    for idx, item in enumerate(analysis_results):
        rank = idx + 1
        item['rank'] = rank
        if item['market'] not in history_db:
            history_db[item['market']] = []
        history_db[item['market']].append({
            "timestamp": time.time(),
            "score": item['score'],
            "rank": rank,
            "price": item['current_price']
        })
        # 히스토리는 최근 24시간 데이터만 보관
        history_db[item['market']] = [h for h in history_db[item['market']] if h['timestamp'] >= time.time() - 86400]

    # 진화형 AI 피드백 로직 (상위 5개 종목 추적 검증 및 가중치 재조정)
    top_5 = analysis_results[:5]
    for top in top_5:
        m = top['market']
        past_records = history_db.get(m, [])
        if len(past_records) >= 2:
            initial_price = past_records[0]['price']
            latest_price = top['current_price']
            perf = (latest_price - initial_price) / (initial_price + 1e-8)
            if perf > 0.01:
                weights["w_pattern"] = min(0.4, weights["w_pattern"] + 0.001)
                weights["w_accumulation"] = min(0.3, weights["w_accumulation"] + 0.001)
            elif perf < -0.01:
                weights["w_pattern"] = max(0.1, weights["w_pattern"] - 0.001)
                weights["w_accumulation"] = max(0.1, weights["w_accumulation"] - 0.001)

    save_json(HISTORY_FILE, history_db)
    save_json(WEIGHTS_FILE, weights)

    # 2. HTML 대시보드 생성
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>업비트 실시간 급등주 포착 대시보드</title>
    <style>
        body {{
            background-color: #f8f9fa;
            color: #333333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .header-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #ffffff;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }}
        .ai-btn {{
            background-color: #007bff;
            color: white;
            padding: 10px 18px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            font-size: 14px;
            transition: background 0.2s;
        }}
        .ai-btn:hover {{
            background-color: #0056b3;
        }}
        .search-box {{
            margin-bottom: 20px;
        }}
        .search-box input {{
            width: 100%;
            padding: 12px 15px;
            font-size: 16px;
            border: 1px solid #ced4da;
            border-radius: 6px;
            box-sizing: border-box;
            outline: none;
            background: #ffffff;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid #e9ecef;
        }}
        th {{
            background-color: #f1f3f5;
            color: #495057;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .plus {{ color: #e03131; font-weight: bold; }}
        .minus {{ color: #1971c2; font-weight: bold; }}
        .top-badge {{
            background: #ffec99;
            color: #e67700;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .update-time {{
            text-align: right;
            font-size: 12px;
            color: #868e96;
            margin-top: 10px;
        }}
        .accumulation {{
            color: #d9480f; 
            font-weight: bold;
        }}
    </style>
    <script>
        function filterTable() {{
            let input = document.getElementById('searchInput').value.toLowerCase();
            let table = document.getElementById('coinTable');
            let tr = table.getElementsByTagName('tr');
            for (let i = 1; i < tr.length; i++) {{
                let tdName = tr[i].getElementsByTagName('td')[1];
                let tdCode = tr[i].getElementsByTagName('td')[0];
                if (tdName || tdCode) {{
                    let textName = tdName.textContent || tdName.innerText;
                    let textCode = tdCode.textContent || tdCode.innerText;
                    if (textName.toLowerCase().indexOf(input) > -1 || textCode.toLowerCase().indexOf(input) > -1) {{
                        tr[i].style.display = "";
                    }} else {{
                        tr[i].style.display = "none";
                    }}
                }}
            }}
        }}
    </script>
</head>
<body>

    <div class="header-container">
        <a href="http://upbit-a.onrender.com" target="_self" class="ai-btn">AI리포트이동</a>
        <h2 style="margin: 0; font-size: 20px; color: #343a40;">🚀 업비트 실시간 급등주 포착 대시보드</h2>
    </div>

    <div class="search-box">
        <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="코인명 또는 티커 검색 (예: 비트코인, KRW-BTC)...">
    </div>

    <table id="coinTable">
        <thead>
            <tr>
                <th>종목코드</th>
                <th>한글코인명</th>
                <th>현재가격 (KRW)</th>
                <th>전일대비등락율</th>
                <th>패턴유사율</th>
                <th>매수우세 (6시간/15분)</th>
                <th>세력매집강도</th>
                <th>예측점수</th>
                <th>최근 3시간 TOP10</th>
            </tr>
        </thead>
        <tbody>
"""

    for item in analysis_results:
        change_class = "plus" if item['change_rate'] > 0 else ("minus" if item['change_rate'] < 0 else "")
        change_sign = "+" if item['change_rate'] > 0 else ""
        rank_badge = f'<span class="top-badge">{item["rank"]}위</span>' if item['rank'] <= 5 else f'{item["rank"]}위'

        html_content += f"""
            <tr>
                <td><b>{item['market']}</b></td>
                <td>{item['name']}</td>
                <td>{item['current_price']:,}</td>
                <td class="{change_class}">{change_sign}{item['change_rate']}%</td>
                <td>{item['pattern_similarity']}%</td>
                <td>{item['positive_count']}회</td>
                <td class="accumulation">{item['accumulation_score']}점</td>
                <td><b>{item['score']}점</b> {rank_badge}</td>
                <td>{item['recent_top10_count']}회</td>
            </tr>
"""

    html_content += f"""
        </tbody>
    </table>

    <div class="update-time">마지막 업데이트 (KST): {current_time_str}</div>

</body>
</html>
"""

    with open(HTML_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("대시보드 HTML 생성 및 자가진화 업데이트 완료.")

if __name__ == "__main__":
    main()
