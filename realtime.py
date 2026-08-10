import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastdtw import fastdtw

# ============================================================
# 기본 설정
# ============================================================

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")

REMOTE_TRACKER_URL = (
    "https://raw.githubusercontent.com/mdog002-wq/upbit/main/"
    "docs/ai_recommend_tracker.json"
)

UPBIT_API_URL = "https://api.upbit.com"

CANDLE_COUNT = 120
MIN_CANDLE_COUNT = 60
MAX_WORKERS = 8
TOP_RESULT_COUNT = 20
REQUEST_TIMEOUT = 5

SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "Upbit-Quant-Bot/2.0",
})

DEFAULT_WEIGHTS = {
    "w_pattern": 0.20,
    "w_vol_cliff": 0.20,
    "w_vol_surge": 0.15,
    "w_ma_alignment": 0.20,
    "w_daily_momentum": 0.10,
    "w_breakout": 0.15,
}

# ============================================================
# 유틸리티
# ============================================================

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def safe_float(value, default=0.0):
    try:
        result = float(value)
        if not np.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default

def clamp(value, minimum=0.0, maximum=100.0):
    value = safe_float(value, minimum)
    return max(minimum, min(maximum, value))

def load_json(filepath, default):
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default

def save_json(filepath, data):
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4, allow_nan=False)
        return True
    except Exception as exc:
        print(f"⚠️ JSON 저장 실패: {filepath} / {exc}")
        return False

def load_weights():
    loaded = load_json(WEIGHTS_FILE, DEFAULT_WEIGHTS.copy())
    if not isinstance(loaded, dict):
        return DEFAULT_WEIGHTS.copy()
    weights = DEFAULT_WEIGHTS.copy()
    for key in weights:
        if key in loaded:
            weights[key] = max(0.0, safe_float(loaded[key], weights[key]))
    return weights

def normalize_weights(weights):
    total = sum(safe_float(value) for value in weights.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {key: value / total for key, value in weights.items()}

# ============================================================
# DTW
# ============================================================

def calculate_dtw_similarity(seq1, seq2):
    try:
        s1 = np.asarray(seq1, dtype=np.float64).reshape(-1)
        s2 = np.asarray(seq2, dtype=np.float64).reshape(-1)
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        if not np.all(np.isfinite(s1)) or not np.all(np.isfinite(s2)):
            return 0.0

        min_len = min(len(s1), len(s2))
        s1, s2 = s1[-min_len:], s2[-min_len:]

        distance, _ = fastdtw(s1, s2, dist=lambda x, y: abs(x - y))
        avg_dist = safe_float(distance) / min_len
        similarity = np.exp(-1.5 * avg_dist) * 100.0
        return round(clamp(similarity), 1)
    except Exception:
        return 0.0

def calculate_max_dtw(seq1, golden_patterns):
    if not isinstance(golden_patterns, list) or not golden_patterns:
        return 0.0
    max_similarity = 0.0
    for pattern in golden_patterns:
        if not isinstance(pattern, (list, tuple, np.ndarray)):
            continue
        similarity = calculate_dtw_similarity(seq1, pattern)
        max_similarity = max(max_similarity, similarity)
    return round(clamp(max_similarity), 1)

# ============================================================
# Upbit API
# ============================================================

def fetch_5m_candles(market, count=CANDLE_COUNT):
    url = f"{UPBIT_API_URL}/v1/candles/minutes/5"
    try:
        response = SESSION.get(url, params={"market": market, "count": count}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"⚠️ {market} 5분봉 조회 실패: {exc}")
        return []

def fetch_all_krw_markets():
    url = f"{UPBIT_API_URL}/v1/market/all"
    try:
        response = SESSION.get(url, params={"isDetails": "false"}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict) and str(item.get("market", "")).startswith("KRW-")]
    except Exception as exc:
        print(f"⚠️ KRW 마켓 조회 실패: {exc}")
        return []

def fetch_remote_recommendations():
    try:
        response = SESSION.get(REMOTE_TRACKER_URL, params={"t": int(time.time())}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or not data:
            return []
        latest = data[-1]
        if not isinstance(latest, dict):
            return []
        coins = latest.get("recommended_coins", [])
        if not isinstance(coins, list):
            return []
        result = [str(coin.get("symbol")) for coin in coins if isinstance(coin, dict) and coin.get("symbol")]
        return list(dict.fromkeys(result))
    except Exception as exc:
        print(f"⚠️ 원격 추천 조회 실패: {exc}")
        return []

def calculate_atr(df, period=14):
    try:
        high = pd.to_numeric(df["high_price"], errors="coerce")
        low = pd.to_numeric(df["low_price"], errors="coerce")
        close = pd.to_numeric(df["trade_price"], errors="coerce")
        previous_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(period, min_periods=period).mean().iloc[-1]
        current_price = safe_float(close.iloc[-1])
        fallback = current_price * 0.015

        if pd.isna(atr):
            return fallback
        return safe_float(atr, fallback)
    except Exception:
        try:
            return safe_float(df["trade_price"].iloc[-1]) * 0.015
        except Exception:
            return 0.0

# ============================================================
# 단일 코인 분석
# ============================================================

def analyze_single_coin(market, korean_name, golden_price_patterns, golden_vol_patterns, weights, recommended_symbols):
    try:
        ticker = str(market).replace("KRW-", "")
        candles = fetch_5m_candles(market)
        if len(candles) < MIN_CANDLE_COUNT:
            return None

        df = pd.DataFrame(candles)
        required_columns = {"timestamp", "trade_price", "prev_closing_price", "candle_acc_trade_volume", "high_price", "low_price"}
        if not required_columns.issubset(df.columns):
            return None

        df = df.sort_values("timestamp").reset_index(drop=True)
        numeric_columns = ["trade_price", "prev_closing_price", "candle_acc_trade_volume", "high_price", "low_price"]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=numeric_columns).reset_index(drop=True)

        if len(df) < MIN_CANDLE_COUNT:
            return None

        current_price = safe_float(df.iloc[-1]["trade_price"])
        previous_close = safe_float(df.iloc[-1]["prev_closing_price"])
        if current_price <= 0 or previous_close <= 0:
            return None

        change_rate = ((current_price - previous_close) / previous_close) * 100.0

        df_2h = df.iloc[-24:].copy().reset_index(drop=True)
        if len(df_2h) < 24:
            return None

        prices = df_2h["trade_price"].to_numpy(dtype=np.float64)
        volumes = df_2h["candle_acc_trade_volume"].to_numpy(dtype=np.float64)

        price_min, price_max = np.min(prices), np.max(prices)
        volume_min, volume_max = np.min(volumes), np.max(volumes)

        price_range = (price_max - price_min) or 1.0
        volume_range = (volume_max - volume_min) or 1.0

        norm_prices = (prices - price_min) / price_range
        norm_volumes = (volumes - volume_min) / volume_range

        price_sim = calculate_max_dtw(norm_prices, golden_price_patterns)
        vol_sim = calculate_max_dtw(norm_volumes, golden_vol_patterns)
        combined_pattern_sim = round(price_sim * 0.7 + vol_sim * 0.3, 1)

        recent_vol = safe_float(df.iloc[-1]["candle_acc_trade_volume"])
        avg_prev_vol = safe_float(df.iloc[-21:-1]["candle_acc_trade_volume"].mean())

        if avg_prev_vol > 0:
            volume_ratio = recent_vol / (avg_prev_vol + 1e-8)
            vol_cliff_score = clamp((1.0 - volume_ratio) * 100.0)
            vol_surge_score = clamp((volume_ratio - 1.0) * 50.0)
        else:
            vol_cliff_score, vol_surge_score = 0.0, 0.0

        df["ma5"] = df["trade_price"].rolling(5, min_periods=5).mean()
        df["ma20"] = df["trade_price"].rolling(20, min_periods=20).mean()
        df["ma60"] = df["trade_price"].rolling(60, min_periods=60).mean()
        last = df.iloc[-1]

        ma5, ma20, ma60 = safe_float(last["ma5"]), safe_float(last["ma20"]), safe_float(last["ma60"])
        if ma5 > ma20 > ma60:
            ma_score = 100.0
        elif ma5 > ma20:
            ma_score = 60.0
        else:
            ma_score = 20.0

        momentum_score = clamp(change_rate * 3.33)
        historical_high = safe_float(df["high_price"].max())
        breakout_score = clamp((current_price / historical_high) * 100.0) if historical_high > 0 else 0.0

        base_score = (
            combined_pattern_sim * weights.get("w_pattern", 0.20) +
            vol_cliff_score * weights.get("w_vol_cliff", 0.20) +
            vol_surge_score * weights.get("w_vol_surge", 0.15) +
            ma_score * weights.get("w_ma_alignment", 0.20) +
            momentum_score * weights.get("w_daily_momentum", 0.10) +
            breakout_score * weights.get("w_breakout", 0.15)
        )

        is_recommended = ticker in recommended_symbols
        if is_recommended:
            base_score += 15.0

        score = round(clamp(base_score), 2)

        atr = calculate_atr(df)
        if atr <= 0:
            atr = current_price * 0.015

        tp1 = current_price + atr * 2.0
        sl = max(0.0, current_price - atr * 1.5)

        return {
            "market": market,
            "ticker": ticker,
            "name": str(korean_name or ticker),
            "current_price": round(current_price, 8),
            "change_rate": round(change_rate, 2),
            "score": score,
            "pattern_similarity": combined_pattern_sim,
            "price_pattern_similarity": price_sim,
            "volume_pattern_similarity": vol_sim,
            "vol_cliff_score": round(vol_cliff_score, 2),
            "vol_surge_score": round(vol_surge_score, 2),
            "ma_score": round(ma_score, 2),
            "momentum_score": round(momentum_score, 2),
            "breakout_score": round(breakout_score, 2),
            "atr": round(atr, 8),
            "tp1": round(tp1, 8),
            "sl": round(sl, 8),
            "is_repo1_recommended": is_recommended,
            "analyzed_at": now_iso(),
        }
    except Exception as exc:
        print(f"⚠️ {market} 분석 오류: {exc}")
        return None

# ============================================================
# Main (GitHub Actions 전용 배치)
# ============================================================

def main():
    print("=" * 70)
    print("🚀 UPBIT QUANT ANALYZER (GitHub Actions)")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)
    weights = normalize_weights(load_weights())
    print("📊 적용 가중치:", weights)

    pattern_data = load_json(PATTERN_FILE, {})
    golden_price_patterns = pattern_data.get("golden_patterns", [])
    golden_vol_patterns = pattern_data.get("golden_volume_patterns", [])

    recommended_symbols = fetch_remote_recommendations()
    markets = fetch_all_krw_markets()

    if not markets:
        print("❌ KRW 마켓을 가져오지 못했습니다.")
        return

    print(f"📋 분석 대상: {len(markets)}개 종목")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                analyze_single_coin,
                m["market"], m.get("korean_name"),
                golden_price_patterns, golden_vol_patterns,
                weights, recommended_symbols
            ): m["market"] for m in markets if m.get("market")
        }

        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    results.sort(key=lambda item: safe_float(item.get("score")), reverse=True)

    if results:
        top_results = results[:TOP_RESULT_COUNT]
        save_json(HISTORY_FILE, top_results)

        print("\n🏆 분석 결과 TOP 10")
        print("-" * 70)
        for rank, r in enumerate(results[:10], start=1):
            mark = " ⭐" if r.get("is_repo1_recommended") else ""
            print(f"{rank:>2}. {r['ticker']:<8} {r['score']:>6.2f}점 | 패턴 {r['pattern_similarity']:>5.1f} | 변동 {r['change_rate']:>6.2f}%{mark}")
        print("-" * 70)
        print("✅ 분석 완료 및 결과 저장 성공.")
    else:
        print("⚠️ 분석 결과가 없습니다.")

if __name__ == "__main__":
    main()
