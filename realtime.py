import os
import io
import time
import json
import requests
import websockets
import asyncio
import pandas as pd
import numpy as np
import smtplib
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import paramiko

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ==============================================================================
# [설정 및 환경변수]
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = "docs"
SCAN_RESULT_JSON = os.path.join(DATA_DIR, "market_scan_result.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
HTML_OUTPUT = os.path.join(DOCS_DIR, "index.html")
EXCEL_FILE_PATH = "업비트_원화마켓_매집_패턴분석_리포트.xlsx"

SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
RECEIVER_EMAILS = [e.strip() for e in os.environ.get("RECEIVER_EMAIL", "").split(",") if e.strip()]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_IDS = [c.strip() for c in os.environ.get("TELEGRAM_CHAT_ID", "").split(",") if c.strip()]

KST = timezone(timedelta(hours=9))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


# ==============================================================================
# [Gemini Structured Output 스키마]
# ==============================================================================
class RecommendedCoin(BaseModel):
    coin_name: str = Field(description="코인 한글명")
    symbol: str = Field(description="티커 심볼")
    reason: str = Field(description="추천 사유")

class AIReportResponse(BaseModel):
    report_markdown: str = Field(description="종합 퀀트 분석 리포트 전문 (마크다운)")
    recommended_coins: list[RecommendedCoin] = Field(description="AI 추천 종목 리스트")


# ==============================================================================
# [웹소켓 및 비트코인 시장 지수 판단]
# ==============================================================================
class UpbitRealtimeWS:
    def __init__(self, markets):
        self.markets = markets
        self.ticker_data = {}
        self.is_running = False

    async def _connect(self):
        url = "wss://api.upbit.com/websocket/v1"
        sub = [{"ticket": "REALTIME"}, {"type": "ticker", "codes": self.markets}]
        while self.is_running:
            try:
                async with websockets.connect(url) as ws:
                    await ws.send(json.dumps(sub))
                    while self.is_running:
                        res = json.loads(await ws.recv())
                        code = res.get("code")
                        if code:
                            self.ticker_data[code] = {
                                "trade_price": res.get("trade_price"),
                                "signed_change_rate": res.get("signed_change_rate", 0) * 100
                            }
            except Exception:
                await asyncio.sleep(2)

    def start(self):
        self.is_running = True
        import threading
        threading.Thread(target=lambda: asyncio.run(self._connect()), daemon=True).start()

def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN: return
    for chat_id in TELEGRAM_CHAT_IDS:
        try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
        except Exception: pass




# ==============================================================================
# [판단 엔진 & 실시간 수급 결합]
# ==============================================================================
def generate_gemini_report(df_top):
    if not GEMINI_API_KEY or df_top.empty:
        return "AI 분석 리포트를 생성할 수 없습니다.", []
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"당신은 암호화폐 퀀트 전문가입니다. 아래 상위 종목 수급 데이터를 판단하고 마크다운 리포트와 최우선 추천 코인을 추출하세요:\n\n{df_top.to_string()}"
        res = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=AIReportResponse)
        )
        data = json.loads(res.text)
        return data.get("report_markdown", ""), data.get("recommended_coins", [])
    except Exception as e:
        return f"AI 리포트 생성 스킵: {e}", []

def process_realtime_decisions():
    if not os.path.exists(SCAN_RESULT_JSON):
        print(f"❌ [오류] 데이터 파일이 존재하지 않습니다: {SCAN_RESULT_JSON}")
        return

    with open(SCAN_RESULT_JSON, "r", encoding="utf-8") as f:
        scan_payload = json.load(f)

    raw_list = scan_payload.get("data", [])
    if not raw_list:
        print("⚠️ [경고] 수집된 코인 데이터가 0건입니다.")
        return

    markets = [item["market"] for item in raw_list]

    # 웹소켓 연결 및 초기 시세 수신 대기 (2초 -> 4초로 변경)
    ws = UpbitRealtimeWS(markets)
    ws.start()
    time.sleep(4) 

    final_results = []
    for item in raw_list:
        m = item["market"]
        ws_info = ws.ticker_data.get(m, {})
        curr_price = ws_info.get("trade_price", item["price"])
        change_rate = ws_info.get("signed_change_rate", 0.0)

        # 실시간 변동률 반영 판단 가중치 적용
        final_score = item["quant_score"] + (change_rate * 0.5)
        final_score = round(max(0.0, min(100.0, final_score)), 1)

        # 1차/2차/Max TP 계산
        tp1 = round(curr_price * 1.03, 2)
        tp2 = round(curr_price * 1.07, 2)
        max_tp = round(curr_price * 1.15, 2)
        sl = round(curr_price * 0.96, 2)

        final_results.append({
            "market": m, "symbol": item["symbol"], "name": item["name"],
            "current_price": curr_price, "change_rate": round(change_rate, 2),
            "score": final_score, "rsi": item["rsi"], "pattern_sim": item["pattern_similarity"],
            "dump_risk": item.get("stgt_dump_risk", item["dump_risk_pct"]),
            "tp1": tp1, "tp2": tp2, "max_tp": max_tp, "sl": sl
        })

    df = pd.DataFrame(final_results).sort_values(by="score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # 급락 덤핑 경보 Telegram 전송
    danger_coins = df[df["dump_risk"] >= 80.0]
    if not danger_coins.empty:
        send_telegram_alert(f"🚨 *[위험 경고]* STGT 덤핑 위험 종목 감지: {', '.join(danger_coins['name'].tolist())}")

    ai_report, rec_coins = generate_gemini_report(df.head(10))

    # HTML 랜더링 및 분출
    render_and_deploy(df, ai_report, rec_coins)

def render_and_deploy(df, ai_report, rec_coins):
    current_time_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

    table_rows = ""
    for _, r in df.iterrows():
        table_rows += f"""
        <tr>
            <td><b>{r['rank']}</b></td>
            <td><b>{r['name']}</b> <span style="font-size:12px; color:#777;">({r['symbol']})</span></td>
            <td>{r['current_price']:,} KRW</td>
            <td style="color:{'#e03131' if r['change_rate']>0 else '#1971c2'}">{r['change_rate']}%</td>
            <td>{r['rsi']}</td>
            <td><b>{r['pattern_sim']}%</b></td>
            <td><b style="color:{'#e03131' if r['dump_risk']>=70 else '#2b8a3e'}">{r['dump_risk']}%</b></td>
            <td><b style="color:#007bff;">{r['score']}점</b></td>
        </tr>"""

    html_template = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Upbit AI Realtime Decision Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body class="bg-light p-4">
    <div class="container-fluid" style="max-width: 1500px;">
        <div class="d-flex justify-content-between align-items-center mb-4 bg-white p-3 rounded shadow-sm">
            <h3 class="fw-bold mb-0">🚀 실시간 AI 암호화폐 수급 판단 대시보드</h3>
            <span class="text-muted">마지막 업데이트: {current_time_str}</span>
        </div>
        <div class="row g-4">
            <div class="col-lg-4">
                <div class="card p-3 shadow-sm mb-3">
                    <h5 class="fw-bold text-primary">🧠 Gemini 종합 리포트</h5>
                    <div id="reportBox" class="small text-secondary" style="max-height: 500px; overflow-y: auto;"></div>
                </div>
            </div>
            <div class="col-lg-8">
                <div class="card p-3 shadow-sm">
                    <h5 class="fw-bold mb-3">🎯 전체 종목 판단 순위</h5>
                    <div style="max-height: 700px; overflow-y: auto;">
                        <table class="table table-hover align-middle text-center">
                            <thead class="table-light sticky-top">
                                <tr><th>순위</th><th>코인명</th><th>현재가</th><th>등락률</th><th>RSI</th><th>패턴유사도</th><th>덤핑위험</th><th>최종점수</th></tr>
                            </thead>
                            <tbody>{table_rows}</tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        document.getElementById("reportBox").innerHTML = marked.parse({json.dumps(ai_report)});
    </script>
</body>
</html>"""

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html_template)

    upload_html_to_oracle_server(HTML_OUTPUT)
    print(f"🎨 [Realtime Engine] 대시보드 및 리포트 갱신 완료 -> `{HTML_OUTPUT}`")

if __name__ == "__main__":
    process_realtime_decisions()
