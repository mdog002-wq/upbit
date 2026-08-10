import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import websockets
from fastdtw import fastdtw


# ============================================================
# 기본 설정
# ============================================================

DATA_DIR = "data"

HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
SIGNAL_HISTORY_FILE = os.path.join(DATA_DIR, "signal_history.json")

REMOTE_TRACKER_URL = (
    "https://raw.githubusercontent.com/mdog002-wq/upbit/main/"
    "docs/ai_recommend_tracker.json"
)

UPBIT_API_URL = "https://api.upbit.com"
UPBIT_WS_URL = "wss://api.upbit.com/websocket/v1"


# ============================================================
# 분석 설정
# ============================================================

CANDLE_COUNT = 120
MIN_CANDLE_COUNT = 60

MAX_WORKERS = 5

TOP_RESULT_COUNT = 20
TOP_WS_COUNT = 10

# 분석 주기
ANALYSIS_INTERVAL_SECONDS = 300

# API timeout
REQUEST_TIMEOUT = 5

# 신호 추적 최대 개수
MAX_SIGNAL_HISTORY = 1000


# ============================================================
# HTTP Session
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "Accept": "application/json",
        "User-Agent": "Upbit-Quant-Bot/2.0",
    }
)


# ============================================================
# 기본 가중치
# ============================================================

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
    """현재 UTC 시간을 ISO 형식으로 반환합니다."""
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_float(value, default=0.0):
    """값을 안전하게 float으로 변환합니다."""
    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def clamp(value, minimum=0.0, maximum=100.0):
    """값을 범위 내로 제한합니다."""
    value = safe_float(value, minimum)

    return max(
        minimum,
        min(maximum, value),
    )


# ============================================================
# JSON
# ============================================================

def load_json(filepath, default):
    """JSON 파일을 안전하게 읽습니다."""

    if not os.path.exists(filepath):
        return default

    try:
        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return default


def save_json(filepath, data):
    """JSON 파일을 안전하게 저장합니다."""

    try:
        directory = os.path.dirname(filepath)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4,
                allow_nan=False,
            )

        return True

    except (
        OSError,
        TypeError,
        ValueError,
    ) as exc:

        print(
            f"⚠️ JSON 저장 실패: "
            f"{filepath} / {exc}"
        )

        return False


# ============================================================
# 가중치
# ============================================================

def load_weights():
    """weights.json을 불러옵니다."""

    loaded = load_json(
        WEIGHTS_FILE,
        DEFAULT_WEIGHTS.copy(),
    )

    if not isinstance(
        loaded,
        dict,
    ):
        return DEFAULT_WEIGHTS.copy()

    weights = DEFAULT_WEIGHTS.copy()

    for key in weights:

        if key in loaded:

            weights[key] = max(
                0.0,
                safe_float(
                    loaded[key],
                    weights[key],
                ),
            )

    return weights


def normalize_weights(weights):
    """
    가중치 합계를 1.0으로 정규화합니다.
    """

    total = sum(
        safe_float(value)
        for value in weights.values()
    )

    if total <= 0:
        return DEFAULT_WEIGHTS.copy()

    return {
        key: value / total
        for key, value in weights.items()
    }


# ============================================================
# DTW
# ============================================================

def calculate_dtw_similarity(seq1, seq2):
    """DTW 기반 패턴 유사도 계산."""

    try:

        s1 = np.asarray(
            seq1,
            dtype=np.float64,
        ).reshape(-1)

        s2 = np.asarray(
            seq2,
            dtype=np.float64,
        ).reshape(-1)

        if len(s1) == 0 or len(s2) == 0:
            return 0.0

        if not np.all(
            np.isfinite(s1)
        ):
            return 0.0

        if not np.all(
            np.isfinite(s2)
        ):
            return 0.0

        min_len = min(
            len(s1),
            len(s2),
        )

        s1 = s1[-min_len:]
        s2 = s2[-min_len:]

        distance, _ = fastdtw(
            s1,
            s2,
            dist=lambda x, y: abs(x - y),
        )

        avg_dist = (
            safe_float(distance)
            / min_len
        )

        similarity = (
            np.exp(-1.5 * avg_dist)
            * 100.0
        )

        return round(
            clamp(similarity),
            1,
        )

    except Exception:
        return 0.0


def calculate_max_dtw(
    seq1,
    golden_patterns,
):
    """여러 골든 패턴 중 최대 유사도."""

    if not isinstance(
        golden_patterns,
        list,
    ):
        return 0.0

    if not golden_patterns:
        return 0.0

    max_similarity = 0.0

    for pattern in golden_patterns:

        if not isinstance(
            pattern,
            (list, tuple, np.ndarray),
        ):
            continue

        similarity = (
            calculate_dtw_similarity(
                seq1,
                pattern,
            )
        )

        max_similarity = max(
            max_similarity,
            similarity,
        )

    return round(
        clamp(max_similarity),
        1,
    )


# ============================================================
# Upbit API
# ============================================================

def fetch_5m_candles(
    market,
    count=CANDLE_COUNT,
):
    """5분봉 조회."""

    url = (
        f"{UPBIT_API_URL}"
        "/v1/candles/minutes/5"
    )

    try:

        response = SESSION.get(
            url,
            params={
                "market": market,
                "count": count,
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        return (
            data
            if isinstance(data, list)
            else []
        )

    except Exception as exc:

        print(
            f"⚠️ {market} "
            f"5분봉 조회 실패: {exc}"
        )

        return []


def fetch_all_krw_markets():
    """KRW 마켓 전체 조회."""

    url = (
        f"{UPBIT_API_URL}"
        "/v1/market/all"
    )

    try:

        response = SESSION.get(
            url,
            params={
                "isDetails": "false",
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(
            data,
            list,
        ):
            return []

        return [
            item
            for item in data
            if isinstance(
                item,
                dict,
            )
            and str(
                item.get(
                    "market",
                    "",
                )
            ).startswith("KRW-")
        ]

    except Exception as exc:

        print(
            f"⚠️ KRW 마켓 조회 실패: {exc}"
        )

        return []


# ============================================================
# 원격 추천
# ============================================================

def fetch_remote_recommendations():
    """원격 추천 코인 조회."""

    try:

        response = SESSION.get(
            REMOTE_TRACKER_URL,
            params={
                "t": int(time.time()),
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(
            data,
            list,
        ):
            return []

        if not data:
            return []

        latest = data[-1]

        if not isinstance(
            latest,
            dict,
        ):
            return []

        coins = latest.get(
            "recommended_coins",
            [],
        )

        if not isinstance(
            coins,
            list,
        ):
            return []

        result = []

        for coin in coins:

            if not isinstance(
                coin,
                dict,
            ):
                continue

            symbol = coin.get(
                "symbol"
            )

            if symbol:
                result.append(
                    str(symbol)
                )

        return list(
            dict.fromkeys(result)
        )

    except Exception as exc:

        print(
            f"⚠️ 원격 추천 조회 실패: {exc}"
        )

        return []


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df,
    period=14,
):
    """ATR 계산."""

    try:

        high = pd.to_numeric(
            df["high_price"],
            errors="coerce",
        )

        low = pd.to_numeric(
            df["low_price"],
            errors="coerce",
        )

        close = pd.to_numeric(
            df["trade_price"],
            errors="coerce",
        )

        previous_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (
                    high
                    - previous_close
                ).abs(),
                (
                    low
                    - previous_close
                ).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = (
            tr
            .rolling(
                period,
                min_periods=period,
            )
            .mean()
            .iloc[-1]
        )

        current_price = safe_float(
            close.iloc[-1]
        )

        fallback = (
            current_price * 0.015
        )

        if pd.isna(atr):
            return fallback

        return safe_float(
            atr,
            fallback,
        )

    except Exception:

        try:

            current_price = safe_float(
                df["trade_price"].iloc[-1]
            )

            return current_price * 0.015

        except Exception:

            return 0.0


# ============================================================
# 단일 코인 분석
# ============================================================

def analyze_single_coin(
    market,
    korean_name,
    golden_price_patterns,
    golden_vol_patterns,
    weights,
    recommended_symbols,
):
    """단일 코인을 분석합니다."""

    try:

        ticker = str(
            market
        ).replace(
            "KRW-",
            "",
        )

        candles = fetch_5m_candles(
            market
        )

        if len(candles) < MIN_CANDLE_COUNT:
            return None

        df = pd.DataFrame(
            candles
        )

        required_columns = {
            "timestamp",
            "trade_price",
            "prev_closing_price",
            "candle_acc_trade_volume",
            "high_price",
            "low_price",
        }

        if not required_columns.issubset(
            df.columns
        ):
            return None

        df = (
            df.sort_values(
                "timestamp"
            )
            .reset_index(drop=True)
        )

        numeric_columns = [
            "trade_price",
            "prev_closing_price",
            "candle_acc_trade_volume",
            "high_price",
            "low_price",
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = (
            df.dropna(
                subset=numeric_columns
            )
            .reset_index(drop=True)
        )

        if len(df) < MIN_CANDLE_COUNT:
            return None

        # ----------------------------------------------------
        # 가격
        # ----------------------------------------------------

        current_price = safe_float(
            df.iloc[-1]["trade_price"]
        )

        previous_close = safe_float(
            df.iloc[-1][
                "prev_closing_price"
            ]
        )

        if (
            current_price <= 0
            or previous_close <= 0
        ):
            return None

        change_rate = (
            (
                current_price
                - previous_close
            )
            / previous_close
        ) * 100.0

        # ----------------------------------------------------
        # 최근 2시간
        # ----------------------------------------------------

        df_2h = (
            df.iloc[-24:]
            .copy()
            .reset_index(drop=True)
        )

        if len(df_2h) < 24:
            return None

        prices = df_2h[
            "trade_price"
        ].to_numpy(
            dtype=np.float64
        )

        volumes = df_2h[
            "candle_acc_trade_volume"
        ].to_numpy(
            dtype=np.float64
        )

        # ----------------------------------------------------
        # 정규화
        # ----------------------------------------------------

        price_min = np.min(
            prices
        )

        price_max = np.max(
            prices
        )

        volume_min = np.min(
            volumes
        )

        volume_max = np.max(
            volumes
        )

        price_range = (
            price_max - price_min
        )

        volume_range = (
            volume_max - volume_min
        )

        if price_range <= 0:
            price_range = 1.0

        if volume_range <= 0:
            volume_range = 1.0

        norm_prices = (
            prices - price_min
        ) / price_range

        norm_volumes = (
            volumes - volume_min
        ) / volume_range

        # ----------------------------------------------------
        # DTW
        # ----------------------------------------------------

        price_sim = calculate_max_dtw(
            norm_prices,
            golden_price_patterns,
        )

        vol_sim = calculate_max_dtw(
            norm_volumes,
            golden_vol_patterns,
        )

        combined_pattern_sim = round(
            price_sim * 0.7
            + vol_sim * 0.3,
            1,
        )

        # ----------------------------------------------------
        # 거래량
        # ----------------------------------------------------

        recent_vol = safe_float(
            df.iloc[-1][
                "candle_acc_trade_volume"
            ]
        )

        avg_prev_vol = safe_float(
            df.iloc[-21:-1][
                "candle_acc_trade_volume"
            ].mean()
        )

        if avg_prev_vol > 0:

            volume_ratio = (
                recent_vol
                / (
                    avg_prev_vol
                    + 1e-8
                )
            )

            # 거래량 급감
            vol_cliff_score = clamp(
                (
                    1.0
                    - volume_ratio
                )
                * 100.0
            )

            # 거래량 급증
            vol_surge_score = clamp(
                (
                    volume_ratio
                    - 1.0
                )
                * 50.0
            )

        else:

            vol_cliff_score = 0.0
            vol_surge_score = 0.0

        # ----------------------------------------------------
        # 이동평균
        # ----------------------------------------------------

        df["ma5"] = (
            df["trade_price"]
            .rolling(
                5,
                min_periods=5,
            )
            .mean()
        )

        df["ma20"] = (
            df["trade_price"]
            .rolling(
                20,
                min_periods=20,
            )
            .mean()
        )

        df["ma60"] = (
            df["trade_price"]
            .rolling(
                60,
                min_periods=60,
            )
            .mean()
        )

        last = df.iloc[-1]

        ma5 = safe_float(
            last["ma5"]
        )

        ma20 = safe_float(
            last["ma20"]
        )

        ma60 = safe_float(
            last["ma60"]
        )

        if ma5 > ma20 > ma60:

            ma_score = 100.0

        elif ma5 > ma20:

            ma_score = 60.0

        else:

            ma_score = 20.0

        # ----------------------------------------------------
        # 모멘텀
        # ----------------------------------------------------

        momentum_score = clamp(
            change_rate * 3.33
        )

        # ----------------------------------------------------
        # 돌파
        # ----------------------------------------------------

        historical_high = safe_float(
            df["high_price"].max()
        )

        if historical_high > 0:

            breakout_score = clamp(
                (
                    current_price
                    / historical_high
                )
                * 100.0
            )

        else:

            breakout_score = 0.0

        # ----------------------------------------------------
        # 최종 점수
        # ----------------------------------------------------

        base_score = (

            combined_pattern_sim
            * weights.get(
                "w_pattern",
                0.20,
            )

            +

            vol_cliff_score
            * weights.get(
                "w_vol_cliff",
                0.20,
            )

            +

            vol_surge_score
            * weights.get(
                "w_vol_surge",
                0.15,
            )

            +

            ma_score
            * weights.get(
                "w_ma_alignment",
                0.20,
            )

            +

            momentum_score
            * weights.get(
                "w_daily_momentum",
                0.10,
            )

            +

            breakout_score
            * weights.get(
                "w_breakout",
                0.15,
            )
        )

        is_recommended = (
            ticker
            in recommended_symbols
        )

        if is_recommended:
            base_score += 15.0

        score = round(
            clamp(base_score),
            2,
        )

        # ----------------------------------------------------
        # ATR
        # ----------------------------------------------------

        atr = calculate_atr(df)

        if atr <= 0:
            atr = (
                current_price
                * 0.015
            )

        tp1 = (
            current_price
            + atr * 2.0
        )

        sl = (
            current_price
            - atr * 1.5
        )

        sl = max(
            0.0,
            sl,
        )

        # ----------------------------------------------------
        # 결과
        # ----------------------------------------------------

        return {
            "market": market,
            "ticker": ticker,
            "name": str(
                korean_name
                or ticker
            ),

            "current_price": round(
                current_price,
                8,
            ),

            "change_rate": round(
                change_rate,
                2,
            ),

            "score": score,

            "pattern_similarity":
                combined_pattern_sim,

            "price_pattern_similarity":
                price_sim,

            "volume_pattern_similarity":
                vol_sim,

            "vol_cliff_score":
                round(
                    vol_cliff_score,
                    2,
                ),

            "vol_surge_score":
                round(
                    vol_surge_score,
                    2,
                ),

            "ma_score":
                round(
                    ma_score,
                    2,
                ),

            "momentum_score":
                round(
                    momentum_score,
                    2,
                ),

            "breakout_score":
                round(
                    breakout_score,
                    2,
                ),

            "atr":
                round(
                    atr,
                    8,
                ),

            "tp1":
                round(
                    tp1,
                    8,
                ),

            "sl":
                round(
                    sl,
                    8,
                ),

            "is_repo1_recommended":
                is_recommended,

            "analyzed_at":
                now_iso(),
        }

    except Exception as exc:

        print(
            f"⚠️ {market} 분석 오류: "
            f"{exc}"
        )

        return None


# ============================================================
# 전체 시장 분석
# ============================================================

def analyze_market():
    """
    전체 KRW 시장을 분석하고
    TOP 결과를 반환합니다.
    """

    print()
    print("=" * 70)
    print("🔍 새로운 시장 분석 시작")
    print("=" * 70)

    weights = normalize_weights(
        load_weights()
    )

    print(
        "📊 적용 가중치:",
        weights,
    )

    # --------------------------------------------------------
    # 패턴
    # --------------------------------------------------------

    pattern_data = load_json(
        PATTERN_FILE,
        {},
    )

    if not isinstance(
        pattern_data,
        dict,
    ):
        pattern_data = {}

    golden_price_patterns = (
        pattern_data.get(
            "golden_patterns",
            [],
        )
    )

    golden_vol_patterns = (
        pattern_data.get(
            "golden_volume_patterns",
            [],
        )
    )

    print(
        f"🧬 가격 패턴: "
        f"{len(golden_price_patterns)}개"
    )

    print(
        f"🧬 거래량 패턴: "
        f"{len(golden_vol_patterns)}개"
    )

    # --------------------------------------------------------
    # 원격 추천
    # --------------------------------------------------------

    recommended_symbols = (
        fetch_remote_recommendations()
    )

    print(
        "🎯 원격 추천:",
        recommended_symbols,
    )

    # --------------------------------------------------------
    # 마켓
    # --------------------------------------------------------

    markets = fetch_all_krw_markets()

    if not markets:

        print(
            "❌ KRW 마켓을 가져오지 못했습니다."
        )

        return []

    print(
        f"📋 분석 대상: "
        f"{len(markets)}개"
    )

    # --------------------------------------------------------
    # 병렬 분석
    # --------------------------------------------------------

    results = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {}

        for market_info in markets:

            market = market_info.get(
                "market"
            )

            korean_name = (
                market_info.get(
                    "korean_name"
                )
            )

            if not market:
                continue

            future = executor.submit(
                analyze_single_coin,
                market,
                korean_name,
                golden_price_patterns,
                golden_vol_patterns,
                weights,
                recommended_symbols,
            )

            futures[future] = market

        completed = 0
        total = len(futures)

        for future in as_completed(
            futures
        ):

            market = futures[future]

            completed += 1

            try:

                result = future.result()

                if result is not None:
                    results.append(
                        result
                    )

            except Exception as exc:

                print(
                    f"⚠️ {market} "
                    f"처리 실패: {exc}"
                )

            if (
                completed % 10 == 0
                or completed == total
            ):

                print(
                    f"📈 진행률 "
                    f"{completed}/{total}"
                )

    # --------------------------------------------------------
    # 정렬
    # --------------------------------------------------------

    results.sort(
        key=lambda item:
            safe_float(
                item.get("score")
            ),
        reverse=True,
    )

    if not results:

        print(
            "❌ 분석 결과가 없습니다."
        )

        return []

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    top_results = results[
        :TOP_RESULT_COUNT
    ]

    save_json(
        HISTORY_FILE,
        top_results,
    )

    # --------------------------------------------------------
    # 출력
    # --------------------------------------------------------

    print()
    print("🏆 TOP 10")
    print("-" * 70)

    for rank, result in enumerate(
        results[:10],
        start=1,
    ):

        mark = (
            " ⭐"
            if result.get(
                "is_repo1_recommended",
                False,
            )
            else ""
        )

        print(
            f"{rank:>2}. "
            f"{result['ticker']:<8} "
            f"{result['score']:>6.2f}점 "
            f"| 패턴 "
            f"{result['pattern_similarity']:>5.1f} "
            f"| 변동 "
            f"{result['change_rate']:>6.2f}%"
            f"{mark}"
        )

    print("-" * 70)

    return top_results


# ============================================================
# 신호 저장
# ============================================================

def load_signal_history():
    """신호 기록을 불러옵니다."""

    data = load_json(
        SIGNAL_HISTORY_FILE,
        [],
    )

    if not isinstance(
        data,
        list,
    ):
        return []

    return data


def save_signal(
    signal,
):
    """새로운 신호를 저장합니다."""

    history = load_signal_history()

    history.append(
        signal
    )

    if len(history) > MAX_SIGNAL_HISTORY:

        history = history[
            -MAX_SIGNAL_HISTORY:
        ]

    save_json(
        SIGNAL_HISTORY_FILE,
        history,
    )


# ============================================================
# 실시간 모니터
# ============================================================

class RealtimeMonitor:

    def __init__(
        self,
        results,
    ):
        self.results = {
            item["market"]: item
            for item in results
        }

        self.running = True

        self.triggered = set()

    def update_results(
        self,
        results,
    ):
        """TOP 결과를 교체합니다."""

        self.results = {
            item["market"]: item
            for item in results
        }

        # 새로운 TOP10이 구성되면
        # 이전 trigger 기록은 초기화
        self.triggered.clear()

    def check_price(
        self,
        market,
        current_price,
    ):
        """현재 가격으로 TP / SL 검사."""

        if market not in self.results:
            return

        item = self.results[
            market
        ]

        ticker = item.get(
            "ticker",
            market,
        )

        tp1 = safe_float(
            item.get("tp1")
        )

        sl = safe_float(
            item.get("sl")
        )

        current_price = safe_float(
            current_price
        )

        if current_price <= 0:
            return

        # ----------------------------------------------------
        # TP1
        # ----------------------------------------------------

        tp_key = (
            f"{market}:TP1"
        )

        if (
            current_price >= tp1
            and tp_key
            not in self.triggered
        ):

            self.triggered.add(
                tp_key
            )

            print()
            print(
                "🎯 TP1 도달!"
            )
            print(
                f"   {ticker}"
            )
            print(
                f"   현재가: "
                f"{current_price}"
            )
            print(
                f"   TP1: "
                f"{tp1}"
            )

            save_signal(
                {
                    "timestamp":
                        now_iso(),
                    "market":
                        market,
                    "ticker":
                        ticker,
                    "event":
                        "TP1",
                    "price":
                        current_price,
                    "target":
                        tp1,
                    "score":
                        item.get(
                            "score"
                        ),
                }
            )

        # ----------------------------------------------------
        # SL
        # ----------------------------------------------------

        sl_key = (
            f"{market}:SL"
        )

        if (
            current_price <= sl
            and sl_key
            not in self.triggered
        ):

            self.triggered.add(
                sl_key
            )

            print()
            print(
                "🛑 SL 도달!"
            )
            print(
                f"   {ticker}"
            )
            print(
                f"   현재가: "
                f"{current_price}"
            )
            print(
                f"   SL: "
                f"{sl}"
            )

            save_signal(
                {
                    "timestamp":
                        now_iso(),
                    "market":
                        market,
                    "ticker":
                        ticker,
                    "event":
                        "SL",
                    "price":
                        current_price,
                    "target":
                        sl,
                    "score":
                        item.get(
                            "score"
                        ),
                }
            )


# ============================================================
# WebSocket
# ============================================================

async def websocket_worker(
    monitor,
):
    """
    WebSocket 실시간 가격 감시.

    monitor.results가 변경되면
    다음 연결부터 새로운 TOP10을 사용합니다.
    """

    reconnect_delay = 3

    while monitor.running:

        markets = list(
            monitor.results.keys()
        )

        if not markets:

            await asyncio.sleep(5)
            continue

        subscribe_data = [
            {
                "ticket":
                    "QUANT_BOT"
            },
            {
                "type":
                    "ticker",
                "codes":
                    markets,
                "isOnlyRealtime":
                    True,
            },
        ]

        try:

            async with websockets.connect(
                UPBIT_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_size=None,
            ) as ws:

                await ws.send(
                    json.dumps(
                        subscribe_data
                    )
                )

                print()
                print(
                    "📡 WebSocket 연결 성공"
                )

                print(
                    "👀 감시:",
                    ", ".join(
                        markets
                    ),
                )

                reconnect_delay = 3

                while monitor.running:

                    try:

                        message = (
                            await asyncio.wait_for(
                                ws.recv(),
                                timeout=30,
                            )
                        )

                    except asyncio.TimeoutError:

                        # 연결이 살아있는지 확인
                        continue

                    if isinstance(
                        message,
                        bytes,
                    ):

                        message = (
                            message.decode(
                                "utf-8"
                            )
                        )

                    try:

                        data = json.loads(
                            message
                        )

                    except (
                        json.JSONDecodeError,
                        TypeError,
                    ):

                        continue

                    if not isinstance(
                        data,
                        dict,
                    ):
                        continue

                    market = data.get(
                        "code"
                    )

                    trade_price = (
                        data.get(
                            "trade_price"
                        )
                    )

                    if (
                        not market
                        or trade_price is None
                    ):
                        continue

                    current_price = (
                        safe_float(
                            trade_price
                        )
                    )

                    monitor.check_price(
                        market,
                        current_price,
                    )

        except asyncio.CancelledError:

            raise

        except Exception as exc:

            print()
            print(
                f"⚠️ WebSocket 오류: "
                f"{exc}"
            )

            print(
                f"🔄 "
                f"{reconnect_delay}초 후 "
                f"재연결..."
            )

            await asyncio.sleep(
                reconnect_delay
            )

            reconnect_delay = min(
                reconnect_delay * 2,
                30,
            )


# ============================================================
# 분석 루프
# ============================================================

async def analysis_loop(
    monitor,
):
    """
    5분마다 전체 시장을 재분석합니다.
    """

    first_run = True

    while monitor.running:

        try:

            # 첫 실행은 즉시
            # 이후에는 5분마다
            if not first_run:

                print()
                print(
                    f"⏳ 다음 분석까지 "
                    f"{ANALYSIS_INTERVAL_SECONDS}초"
                )

                await asyncio.sleep(
                    ANALYSIS_INTERVAL_SECONDS
                )

            first_run = False

            # ------------------------------------------------
            # 시장 분석
            # ------------------------------------------------

            results = await asyncio.to_thread(
                analyze_market
            )

            if not results:

                print(
                    "⚠️ 분석 결과가 없어 "
                    "다음 사이클로 넘어갑니다."
                )

                continue

            # ------------------------------------------------
            # TOP10 갱신
            # ------------------------------------------------

            monitor.update_results(
                results[
                    :TOP_WS_COUNT
                ]
            )

            print()
            print(
                "🔄 실시간 감시 대상 갱신 완료"
            )

            print(
                "📡",
                ", ".join(
                    monitor.results.keys()
                ),
            )

        except asyncio.CancelledError:

            raise

        except Exception as exc:

            print(
                f"⚠️ 분석 루프 오류: "
                f"{exc}"
            )

            await asyncio.sleep(
                10
            )


# ============================================================
# WebSocket + 분석 통합 실행
# ============================================================

async def run_bot():
    """전체 자동화 봇을 실행합니다."""

    print()
    print("=" * 70)
    print(
        "🚀 UPBIT QUANT BOT 2.0"
    )
    print(
        "🧬 DTW + 거래량 + MA + 모멘텀"
    )
    print(
        "📡 실시간 WebSocket + 자동 재분석"
    )
    print("=" * 70)

    os.makedirs(
        DATA_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 최초 분석
    # --------------------------------------------------------

    initial_results = await asyncio.to_thread(
        analyze_market
    )

    if not initial_results:

        print(
            "❌ 최초 분석에 실패했습니다."
        )

        return

    monitor = RealtimeMonitor(
        initial_results[
            :TOP_WS_COUNT
        ]
    )

    # --------------------------------------------------------
    # 병렬 실행
    # --------------------------------------------------------

    websocket_task = asyncio.create_task(
        websocket_worker(
            monitor
        )
    )

    analysis_task = asyncio.create_task(
        analysis_loop(
            monitor
        )
    )

    try:

        await asyncio.gather(
            websocket_task,
            analysis_task,
        )

    except asyncio.CancelledError:

        print(
            "🛑 봇 종료 요청"
        )

        monitor.running = False

        websocket_task.cancel()
        analysis_task.cancel()

        await asyncio.gather(
            websocket_task,
            analysis_task,
            return_exceptions=True,
        )

        raise

    except Exception as exc:

        print(
            f"❌ 메인 실행 오류: "
            f"{exc}"
        )

        monitor.running = False

        websocket_task.cancel()
        analysis_task.cancel()

        await asyncio.gather(
            websocket_task,
            analysis_task,
            return_exceptions=True,
        )


# ============================================================
# Main
# ============================================================

def main():

    try:

        asyncio.run(
            run_bot()
        )

    except KeyboardInterrupt:

        print()
        print(
            "🛑 프로그램을 종료했습니다."
        )

    except Exception as exc:

        print()
        print(
            f"❌ 치명적 오류: {exc}"
        )


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()
