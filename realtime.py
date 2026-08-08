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

def fetch_candles(market, count=672):
    """최근 1주일간의 15분봉 데이터 수집 (672개 = 7일 * 24시간 * 4회)"""
    all_candles = []
    to_param = ""
    while len(all_candles) < count:
        req_count = min(200, count - len(all_candles))
        url = f"https://api.upbit.com/v1/candles/minutes/15?market={market}&count={req_count}"
        if to_param:
            url += f"&to={to_param}"
        res = requests.get(url)
        if res.status_code != 200:
            break
        data = res.json()
        if not data:
            break
        all_candles.extend(data)
        if len(data) < req_count:
            break
        to_param = data[-1]['candle_date_time_utc']
        time.sleep(0.03) # API 호출 제한 준수
    return all_candles

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

    # 1. 초기 데이터 및 가중치 불러오기
    history_db = load_json(HISTORY_FILE, {}) 
    weights = load_json(WEIGHTS_FILE, {
        "w_pattern": 0.25,
        "w_buy_sell": 0.25,
        "w_recent_rank": 0.15,
        "w_ai_volatility": 0.15,
        "w_accumulation": 0.20
    })

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
        ticker = market.replace("KRW-", "")
        
        # 1주일 치 15분봉 수집 (672개)
        candles = fetch_candles(market, count=672)
        if len(candles) < 24:
            continue

        df = pd.DataFrame(candles)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 최근 6시간 분석용 캔들 (마지막 24개)
        df_6h = df.iloc[-24:].copy().reset_index(drop=True)

        current_price = df_6h.iloc[-1]['trade_price']
        prev_close = df_6h.iloc[0]['opening_price']
        change_rate = ((current_price - prev_close) / prev_close) * 100

        # 최근 6시간 매수 우세 횟수 (양봉 캔들)
        positive_count = sum(1 for _, row in df_6h.iterrows() if row['trade_price'] > row['opening_price'])

        # 패턴 유사도 계산
        prices = df_6h['trade_price'].values
        price_min, price_max = prices.min(), prices.max()
        norm_prices = (prices - price_min) / (price_max - price_min + 1e-8)
        distance = np.linalg.norm(norm_prices - ideal_pattern)
        pattern_similarity = max(0.0, float(1.0 - (distance / np.sqrt(len(norm_prices))))) * 100

        # AI 변동성 점수
        volume_std = df_6h['candle_acc_trade_volume'].std()
        volume_mean = df_6h['candle_acc_trade_volume'].mean()
        ai_volatility_score = float(min(100.0, (volume_std / (volume_mean + 1e-8)) * 50))

        # 세력 매집 추적 점수
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

        # 지난 1주일간 15분 내 5% 이상 변동 집계
        up_5pct_count = sum(1 for _, row in df.iterrows() if ((row['high_price'] - row['opening_price']) / (row['opening_price'] + 1e-8)) * 100 >= 5.0)
        down_5pct_count = sum(1 for _, row in df.iterrows() if ((row['opening_price'] - row['low_price']) / (row['opening_price'] + 1e-8)) * 100 >= 5.0)

        # 유동성 지수
        df_24h = df.iloc[-96:] if len(df) >= 96 else df
        acc_24h_krw = df_24h['candle_acc_trade_price'].sum()
        if acc_24h_krw > 0:
            liquidity_index = round(min(100.0, max(0.0, (np.log10(acc_24h_krw) - 7) * 20)), 1)
        else:
            liquidity_index = 0.0

        # 최근 3시간 TOP10 유지 횟수
        market_history = history_db.get(market, [])
        now_ts = time.time()
        three_hours_ago = now_ts - 3 * 3600
        recent_top10_count = sum(1 for h in market_history if h['timestamp'] >= three_hours_ago and h['rank'] <= 10)

        # 가중치 기반 최종 예측 점수
        score = (
            pattern_similarity * weights["w_pattern"] +
            (positive_count / 24.0 * 100) * weights["w_buy_sell"] +
            min(100.0, recent_top10_count * 20) * weights["w_recent_rank"] +
            ai_volatility_score * weights["w_ai_volatility"] +
            accumulation_score * weights["w_accumulation"]
        )

        analysis_results.append({
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
        })

    # 예측 점수 정렬
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
        history_db[item['market']] = [h for h in history_db[item['market']] if h['timestamp'] >= time.time() - 86400]

    # 가중치 피드백 재조정
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

    # JavaScript용 분석 데이터 JSON
    js_data_json = json.dumps(analysis_results, ensure_ascii=False)

    # 2. 테이블 행(Rows) HTML 생성
    table_rows_html = ""
    for item in analysis_results:
        change_class = "plus" if item['change_rate'] > 0 else ("minus" if item['change_rate'] < 0 else "")
        change_sign = "+" if item['change_rate'] > 0 else ""

        table_rows_html += f"""
            <tr onclick="openModal('{item["market"]}')">
                <td data-val="{item['rank']}"><b>{item['rank']}</b></td>
                <td data-val="{item['name']}">
                    <b>{item['name']}</b> <span class="ticker-symbol">({item['ticker']})</span>
                </td>
                <td data-val="{item['current_price']}">{item['current_price']:,}</td>
                <td data-val="{item['change_rate']}" class="{change_class}">{change_sign}{item['change_rate']}%</td>
                <td data-val="{item['pattern_similarity']}">{item['pattern_similarity']}%</td>
                <td data-val="{item['accumulation_score']}" class="accumulation">{item['accumulation_score']}점</td>
                <td data-val="{item['liquidity_index']}" class="liquidity">{item['liquidity_index']}점</td>
                <td data-val="{item['recent_top10_count']}">{item['recent_top10_count']}회</td>
                <td data-val="{item['positive_count']}">{item['positive_count']}회</td>
                <td data-val="{item['up_5pct_count']}"><span class="plus">▲{item['up_5pct_count']}회</span> / <span class="minus">▼{item['down_5pct_count']}회</span></td>
                <td data-val="{item['score']}"><b>{item['score']}점</b></td>
            </tr>
        """

    # 3. 전체 HTML 템플릿 작성
    html_template = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>업비트 실시간 급등주 포착 대시보드</title>
    <style>
        body {
            background-color: #f8f9fa;
            color: #333333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
        }
        .header-container {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            background: #ffffff;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .header-left { text-align: left; }
        .header-center { text-align: center; }
        .header-right {
            text-align: right;
            font-size: 13px;
            color: #495057;
            font-weight: 500;
        }
        .ai-btn {
            background-color: #007bff;
            color: white;
            padding: 10px 18px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            font-size: 14px;
            display: inline-block;
            transition: background 0.2s;
        }
        .ai-btn:hover { background-color: #0056b3; }
        .search-box { margin-bottom: 20px; }
        .search-box input {
            width: 100%;
            padding: 12px 15px;
            font-size: 16px;
            border: 1px solid #ced4da;
            border-radius: 6px;
            box-sizing: border-box;
            outline: none;
            background: #ffffff;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        th, td {
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid #e9ecef;
        }
        th {
            background-color: #f1f3f5;
            color: #495057;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
            transition: background-color 0.2s;
        }
        th:hover { background-color: #e9ecef; }
        th::after {
            content: " ↕";
            font-size: 11px;
            color: #adb5bd;
        }
        tbody tr {
            cursor: pointer;
            transition: background-color 0.15s;
        }
        tbody tr:hover { background-color: #e9ecef !important; }
        .plus { color: #e03131; font-weight: bold; }
        .minus { color: #1971c2; font-weight: bold; }
        .ticker-symbol {
            font-size: 12px;
            color: #868e96;
            font-weight: normal;
            margin-left: 4px;
        }
        .accumulation { color: #d9480f; font-weight: bold; }
        .liquidity { color: #2b8a3e; font-weight: bold; }

        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.4);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal-box {
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            width: 350px;
            height: 450px;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.25);
            overflow: hidden;
        }
        .modal-box iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        .detail-card {
            padding: 15px;
            box-sizing: border-box;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            background: #ffffff;
            font-size: 13px;
        }
        .detail-header {
            border-bottom: 2px solid #007bff;
            padding-bottom: 8px;
            margin-bottom: 10px;
        }
        .detail-title {
            font-size: 18px;
            font-weight: bold;
            color: #212529;
            margin: 0;
        }
        .detail-row {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #f1f3f5;
        }
        .detail-label { color: #495057; font-weight: 500; }
        .detail-value { font-weight: bold; }
    </style>
    <script>
        const analysisData = __JS_DATA_JSON__;
        let sortDirections = {};

        window.addEventListener('DOMContentLoaded', () => {
            const urlParams = new URLSearchParams(window.location.search);
            const symbol = urlParams.get('symbol');

            if (symbol) {
                const targetMarket = symbol.toUpperCase().startsWith('KRW-') ? symbol.toUpperCase() : `KRW-${symbol.toUpperCase()}`;
                const item = analysisData.find(d => d.market === targetMarket || d.ticker === symbol.toUpperCase());
                
                if (item) {
                    const changeClass = item.change_rate > 0 ? "plus" : (item.change_rate < 0 ? "minus" : "");
                    const changeSign = item.change_rate > 0 ? "+" : "";
                    
                    document.body.innerHTML = `
                        <div style="padding: 15px; font-family: sans-serif; background: #fff; height: 100%; box-sizing: border-box;">
                          <h6 style="margin-top:0; margin-bottom: 12px; font-weight: bold; color: #007bff;">⚡ AI실시간 </h6>
                          <div style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom: 1px solid #eee; font-size: 13px;"><span class="detail-label">현재 가격</span><span class="detail-value">${item.current_price.toLocaleString()} KRW</span></div>
                          <div style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom: 1px solid #eee; font-size: 13px;"><span class="detail-label">전일 대비</span><span class="detail-value ${changeClass}">${changeSign}${item.change_rate}%</span></div>
                          <div style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom: 1px solid #eee; font-size: 13px;"><span class="detail-label">예측 점수</span><span class="detail-value" style="color:#007bff;">${item.score}점 (${item.rank}위)</span></div>
                          <div style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom: 1px solid #eee; font-size: 13px;"><span class="detail-label">패턴 유사율</span><span class="detail-value">${item.pattern_similarity}%</span></div>
                          <div style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom: 1px solid #eee; font-size: 13px;"><span class="detail-label">세력 매집 강도</span><span class="detail-value accumulation">${item.accumulation_score}점</span></div>
                          <div style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom: 1px solid #eee; font-size: 13px;"><span class="detail-label">유동성 지수</span><span class="detail-value liquidity">${item.liquidity_index}점</span></div>
                          <div style="display:flex; justify-content:space-between; padding: 6px 0; font-size: 13px;"><span class="detail-label">1주일 5% 변동</span><span class="detail-value"><span class="plus">▲${item.up_5pct_count}</span> / <span class="minus">▼${item.down_5pct_count}</span></span></div>
                        </div>
                    `;
                    document.body.style.padding = '0';
                    document.body.style.background = '#fff';
                }
            }
        });

        function filterTable() {
            let input = document.getElementById('searchInput').value.toLowerCase();
            let table = document.getElementById('coinTable');
            let tr = table.getElementsByTagName('tr');
            for (let i = 1; i < tr.length; i++) {
                let tdName = tr[i].getElementsByTagName('td')[1];
                if (tdName) {
                    let textName = tdName.textContent || tdName.innerText;
                    if (textName.toLowerCase().indexOf(input) > -1) {
                        tr[i].style.display = "";
                    } else {
                        tr[i].style.display = "none";
                    }
                }
            }
        }

        function sortTable(columnIndex) {
            let table = document.getElementById('coinTable');
            let tbody = table.querySelector('tbody');
            let rows = Array.from(tbody.querySelectorAll('tr'));

            if (!(columnIndex in sortDirections)) {
                sortDirections[columnIndex] = (columnIndex === 0 || columnIndex === 1) ? true : false;
            } else {
                sortDirections[columnIndex] = !sortDirections[columnIndex];
            }

            let isAscending = sortDirections[columnIndex];

            rows.sort((a, b) => {
                let valA = a.children[columnIndex].getAttribute('data-val') || a.children[columnIndex].innerText.trim();
                let valB = b.children[columnIndex].getAttribute('data-val') || b.children[columnIndex].innerText.trim();

                let numA = parseFloat(valA.replace(/[^0-9.-]/g, ''));
                let numB = parseFloat(valB.replace(/[^0-9.-]/g, ''));

                if (!isNaN(numA) && !isNaN(numB)) {
                    return isAscending ? numA - numB : numB - numA;
                } else {
                    return isAscending ? valA.localeCompare(valB) : valB.localeCompare(valA);
                }
            });

            rows.forEach(row => tbody.appendChild(row));
        }
    </script>
</head>
<body>

    <div class="header-container">
        <div class="header-left">
            <a href="https://upbit-a.onrender.com" target="_self" class="ai-btn">AI리포트이동</a>
        </div>
        <div class="header-center">
            <h2 style="margin: 0; font-size: 20px; color: #343a40;">🚀 업비트 실시간 급등주 포착 대시보드</h2>
        </div>
        <div class="header-right">
            마지막 업데이트 (KST): <b>__CURRENT_TIME_STR__</b>
        </div>
    </div>

    <div class="search-box">
        <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="코인명 또는 티커 검색 (예: 비트코인, BTC)...">
    </div>

    <table id="coinTable">
        <thead>
            <tr>
                <th onclick="sortTable(0)">순위</th>
                <th onclick="sortTable(1)">한글코인명</th>
                <th onclick="sortTable(2)">현재가격 (KRW)</th>
                <th onclick="sortTable(3)">전일대비등락율</th>
                <th onclick="sortTable(4)">패턴유사율</th>
                <th onclick="sortTable(5)">세력매집강도</th>
                <th onclick="sortTable(6)">유동성지수</th>
                <th onclick="sortTable(7)">최근 3시간 TOP10</th>
                <th onclick="sortTable(8)">매수우세 (6시간/15분)</th>
                <th onclick="sortTable(9)">1주일 5% 변동 (15분봉)</th>
                <th onclick="sortTable(10)">예측점수</th>
            </tr>
        </thead>
        <tbody>
            __TABLE_ROWS_HTML__
        </tbody>
    </table>

    <!-- 모달 창 -->
    <div id="coinDetailModal" class="modal-overlay" onclick="closeModal()">
      <div class="modal-box" onclick="event.stopPropagation();" style="width: 820px; max-width: 95vw; height: 500px; padding: 20px; display: flex; flex-direction: column;">
        
        <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; margin-bottom: 15px; border-bottom: 1px solid #dee2e6;">
          <h4 id="rModalTitle" style="margin: 0; font-size: 18px; font-weight: bold;">종목 상세 정보</h4>
          <button onclick="closeModal()" style="border:none; background:none; font-size: 20px; cursor:pointer;">✕</button>
        </div>

        <div style="display: flex; gap: 15px; flex: 1; min-height: 0;">
          <!-- 좌측: R 사이트 AI실시간 상세 정보 -->
          <div style="flex: 1; background: #f8f9fa; border-radius: 8px; padding: 15px; overflow-y: auto;">
            <h5 style="margin-top:0; margin-bottom: 12px; font-size: 14px; color: #007bff; font-weight: bold;">⚡ AI실시간 </h5>
            <div id="rModalContentR"></div>
          </div>

          <!-- 우측: 상단(iframe A) / 하단(A추천종목 R순위 정렬 리스트) -->
          <div style="flex: 1; display: flex; flex-direction: column; gap: 10px;">
            <!-- 우측 상단: A 사이트 iframe -->
            <div style="flex: 1; border: 1px solid #dee2e6; border-radius: 8px; overflow: hidden; min-height: 180px;">
              <iframe id="modalIframeA" src="" style="width: 100%; height: 100%; border: none;"></iframe>
            </div>

            <!-- 우측 하단: A 사이트 추천 종목 (R 사이트 순위순 정렬) -->
            <div style="height: 190px; border: 1px solid #dee2e6; border-radius: 8px; padding: 10px; background: #ffffff; overflow-y: auto;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 13px; font-weight: bold; color: #212529;">🎯 AI 추천 종목</span>
                <span style="font-size: 11px; color: #6c757d;">R 사이트 순위순 정렬</span>
              </div>
              <div id="modalRecommendList" style="font-size: 12px;">
                <div style="text-align: center; color: #6c757d; padding: 15px 0;">순위 데이터 연동 중...</div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>

    <script>
    // A 사이트의 추천 종목 데이터를 가져오는 함수
    async function fetchARecommendations() {
      try {
        const response = await fetch('https://upbit-a.onrender.com/api/rankings'); // A 사이트 추천 API/JSON
        if (!response.ok) throw new Error("A 사이트 연동 실패");
        return await response.json();
      } catch (e) {
        console.warn("A 사이트 연동 실패, 기본 데이터로 대체합니다.", e);
        return [];
      }
    }

    // A 사이트 추천 종목 중 모니터에 포함된 코인들의 R 사이트 순위 정렬 표출
    async function renderRecommendedListSortedByR() {
      const container = document.getElementById('modalRecommendList');
      const aItems = await fetchARecommendations();

      if (!aItems || aItems.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #6c757d; padding: 10px;">추천 종목을 불러올 수 없습니다.</div>';
        return;
      }

      // A 사이트 추천 종목 코인들에 R 사이트 현재 순위 매핑
      const mappedList = aItems.map(aCoin => {
        const marketKey = aCoin.market || (aCoin.symbol ? `KRW-${aCoin.symbol}` : '');
        const tickerKey = aCoin.ticker || aCoin.symbol;

        // R 사이트(analysisData)에서 동일 코인 검색
        const rMatch = analysisData.find(r => 
          r.market === marketKey || 
          r.ticker === tickerKey ||
          r.market === `KRW-${tickerKey}`
        );

        return {
          name: aCoin.name || (rMatch ? rMatch.name : tickerKey),
          ticker: tickerKey,
          market: marketKey,
          score: rMatch ? rMatch.score : (aCoin.score || 0),
          // R 사이트 순위 (없을 시 9999)
          r_rank: rMatch && rMatch.rank ? parseInt(rMatch.rank, 10) : 9999
        };
      });

      // R 사이트 순위(r_rank) 오름차순 정렬 (1위 -> 2위...)
      mappedList.sort((a, b) => a.r_rank - b.r_rank);

      // HTML 랜더링
      let html = '<table style="width: 100%; border-collapse: collapse; text-align: center;">';
      html += '<thead style="background: #f8f9fa; border-bottom: 1px solid #dee2e6;">' +
              '<tr><th style="padding: 4px;">R순위</th><th style="padding: 4px; text-align: left;">종목명</th><th style="padding: 4px; text-align: right;">R점수</th></tr>' +
              '</thead><tbody>';

      mappedList.forEach(coin => {
        const rankDisplay = coin.r_rank !== 9999 
          ? `<b style="color: #007bff;">${coin.r_rank}위</b>` 
          : `<span style="color: #adb5bd;">-</span>`;

        html += `
          <tr onclick="openModal('${coin.market}')" style="cursor: pointer; border-bottom: 1px solid #f1f3f5;">
            <td style="padding: 5px;">${rankDisplay}</td>
            <td style="padding: 5px; text-align: left;">
              <b>${coin.name}</b> <span style="font-size: 10px; color: #868e96;">(${coin.ticker})</span>
            </td>
            <td style="padding: 5px; text-align: right; font-weight: bold; color: #2b8a3e;">
              ${coin.score}점
            </td>
          </tr>
        `;
      });

      html += '</tbody></table>';
      container.innerHTML = html;
    }

    function openModal(symbol) {
      const targetMarket = symbol.toUpperCase().startsWith('KRW-') ? symbol.toUpperCase() : `KRW-${symbol.toUpperCase()}`;
      const item = analysisData.find(d => d.market === targetMarket || d.ticker === symbol.toUpperCase());
      const titleEl = document.getElementById('rModalTitle');
      const contentR = document.getElementById('rModalContentR');
      const iframeA = document.getElementById('modalIframeA');

      if (item) {
        const changeClass = item.change_rate > 0 ? "plus" : (item.change_rate < 0 ? "minus" : "");
        const changeSign = item.change_rate > 0 ? "+" : "";

        titleEl.innerText = `${item.name} (${item.ticker})`;
        contentR.innerHTML = `
          <div class="detail-row" style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom:1px solid #eee; font-size: 13px;"><span class="detail-label">현재 가격</span><span class="detail-value">${item.current_price.toLocaleString()} KRW</span></div>
          <div class="detail-row" style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom:1px solid #eee; font-size: 13px;"><span class="detail-label">전일 대비</span><span class="detail-value ${changeClass}">${changeSign}${item.change_rate}%</span></div>
          <div class="detail-row" style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom:1px solid #eee; font-size: 13px;"><span class="detail-label">예측 점수</span><span class="detail-value" style="color:#007bff;">${item.score}점 (${item.rank}위)</span></div>
          <div class="detail-row" style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom:1px solid #eee; font-size: 13px;"><span class="detail-label">패턴 유사율</span><span class="detail-value">${item.pattern_similarity}%</span></div>
          <div class="detail-row" style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom:1px solid #eee; font-size: 13px;"><span class="detail-label">세력 매집 강도</span><span class="detail-value accumulation">${item.accumulation_score}점</span></div>
          <div class="detail-row" style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom:1px solid #eee; font-size: 13px;"><span class="detail-label">유동성 지수</span><span class="detail-value liquidity">${item.liquidity_index}점</span></div>
          <div class="detail-row" style="display:flex; justify-content:space-between; padding: 6px 0; font-size: 13px;"><span class="detail-label">1주일 5% 변동</span><span class="detail-value"><span class="plus">▲${item.up_5pct_count}</span> / <span class="minus">▼${item.down_5pct_count}</span></span></div>
        `;
        
        iframeA.src = `https://upbit-a.onrender.com/?symbol=${item.market}`;
      }

      // 모달이 열릴 때 A사이트 추천 코인의 R사이트 순위 정렬 리스트 랜더링
      renderRecommendedListSortedByR();

      document.getElementById('coinDetailModal').style.display = 'flex';
    }

    function closeModal() {
      document.getElementById('coinDetailModal').style.display = 'none';
      document.getElementById('modalIframeA').src = '';
    }
    </script>
</body>
</html>
"""

    # 4. 플레이스홀더를 실제 데이터로 안전하게 치환
    html_content = html_template.replace("__JS_DATA_JSON__", js_data_json)
    html_content = html_content.replace("__CURRENT_TIME_STR__", current_time_str)
    html_content = html_content.replace("__TABLE_ROWS_HTML__", table_rows_html)

    with open(HTML_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("대시보드 HTML 생성 및 자가진화 업데이트 완료.")

if __name__ == "__main__":
    main()
