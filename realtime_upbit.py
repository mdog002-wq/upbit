import os
import json
import time
from datetime import datetime
import sqlite3
import requests
import pandas as pd

# 💡 경로 설정: GitHub 저장소 루트와 docs 폴더
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
WEIGHTS_FILE = os.path.join(BASE_DIR, "strategy_weights.json")
DB_FILE = os.path.join(BASE_DIR, "upbit_history.db")
HTML_FILE = os.path.join(DOCS_DIR, "index.html")

# 폴더가 없으면 생성
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# (이전 코드의 DB, 로직 함수들은 유지)
# ... [init_db, save_to_db, load_weights, save_weights, evolve_weights, UpbitEvolutionAnalyzer 등 동일] ...

def update_dashboard_html(records, generation):
    """분석 데이터를 바탕으로 docs/index.html 파일을 업데이트"""
    data_json = json.dumps({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "generation": generation, "results": records})
    
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>업비트 AI 자가진화 대시보드</title>
    <style>
        body {{ font-family: 'Pretendard', sans-serif; background-color: #f4f6f9; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 20px; color: #093687; }}
        table {{ width: 100%; border-collapse: collapse; background: white; }}
        th, td {{ padding: 10px; border: 1px solid #ddd; text-align: center; }}
        .plus {{ color: #c84a31; }} .minus {{ color: #1261c4; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 업비트 AI 실시간 분석</h1>
        <p id="update-info">데이터 로딩 중...</p>
    </div>
    <table id="data-table">
        <thead><tr><th>순위</th><th>코인명</th><th>현재가격</th><th>등락율</th><th>최종점수</th></tr></thead>
        <tbody id="table-body"></tbody>
    </table>
    <script>
        const data = {data_json};
        document.getElementById('update-info').innerText = "마지막 업데이트: " + data.time + " | 세대: " + data.generation;
        let html = "";
        data.results.forEach((row, i) => {{
            html += `<tr><td>${{i+1}}</td><td>${{row.korean_name}}</td><td>${{row.trade_price.toLocaleString()}}원</td><td class="${{row.signed_change_rate >= 0 ? 'plus' : 'minus'}}">${{row.signed_change_rate}}%</td><td>${{row.score}}</td></tr>`;
        }});
        document.getElementById('table-body').innerHTML = html;
    </script>
</body>
</html>"""
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)

def run_analysis_process():
    # ... [기존 분석 로직 실행] ...
    # ... records, evolved_weights 반환 후 ...
    update_dashboard_html(records, evolved_weights['generation'])
    return records, evolved_weights['generation']

if __name__ == "__main__":
    run_analysis_process()
