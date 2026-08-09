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
# 1. AI 추천 종목 파싱 (전공정 이전 완벽 복원)
# ==========================================
def fetch_ai_recommendations():
    """upbit-a 레포지토리 docs/ai_recommend_tracker.json에서 AI 추천 티커 추출"""
    refined_set = set()

    def parse_item(item):
        if isinstance(item, str):
            match = re.search(r"\(([A-Z0-9]+)\)", item.upper())
            ticker = match.group(1) if match else item.strip().upper().replace("KRW-", "")
            if ticker and len(ticker) <= 10:
