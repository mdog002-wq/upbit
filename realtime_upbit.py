from datetime import datetime
import json
import os
import sqlite3
import pandas as pd
import requests

# 💡 경로 설정: GitHub 저장소 루트와 docs 폴더
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
WEIGHTS_FILE = os.path.join(BASE_DIR, "strategy_weights.json")
DB_FILE = os.path.join(BASE_DIR, "upbit_history.db")
HTML_FILE = os.path.join(DOCS_DIR, "index.html")

# 폴더가 없으면 생성
if not os.path.exists(DOCS_DIR):
  os.makedirs(DOCS_DIR)


def update_dashboard_html(records, generation):
  """분석 데이터를 바탕으로 docs/index.html 파일을 업데이트"""
  data_json = json.dumps({
      "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "generation": generation,
      "results": records,
  })

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
  """기존 분석 로직 실행 및 데이터 정의"""
  # 예시용 더미 데이터 구조 (실제 분석 로직 결과물로 대체해야 합니다)
  # 만약 기존에 사용하던 분석기 클래스나 함수가 있다면 이 부분을 해당 로직으로 채워넣으세요.
  evolved_weights = {"generation": 1}

  # 예시 records 데이터 (실제 수집된 코인 분석 결과 딕셔너리 리스트)
  records = [
      {
          "korean_name": "비트코인",
          "trade_price": 95000000,
          "signed_change_rate": 2.5,
          "score": 88.5,
      },
      {
          "korean_name": "이더리움",
          "trade_price": 3500000,
          "signed_change_rate": -1.2,
          "score": 75.0,
      },
  ]

  # 💡 필수: update_dashboard_html 호출 전에 records와 evolved_weights가 반드시 정의되어야 합니다.
  update_dashboard_html(records, evolved_weights["generation"])
  return records, evolved_weights["generation"]


if __name__ == "__main__":
  run_analysis_process()
