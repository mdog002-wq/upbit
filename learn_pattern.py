import json
import os
import time
import numpy as np
import pandas as pd
import requests

DATA_DIR = "data"
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")


def get_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            markets = res.json()
            return [
                m["market"] for m in markets if m["market"].startswith("KRW-")
            ]
    except Exception as e:
        print(f"⚠️ 마켓 목록 조회 실패: {e}")
    return []


def fetch_5m_candles_deep(market, target_count=2000):
    """5분 봉 데이터 대량 수집 (Upbit API 429 요청 제한 안전 대응 적용)"""
    all_candles = []
    to_param = ""
    retry_count = 0

    while len(all_candles) < target_count:
        req_count = min(200, target_count - len(all_candles))
        url = f"https://api.upbit.com/v1/candles/minutes/5?market={market}&count={req_count}"
        if to_param:
            url += f"&to={to_param}"

        try:
            res = requests.get(url, timeout=5)

            # [개선 1] API 요청 제한(429) 발생 시 대기 후 재시도
            if res.status_code == 429:
                time.sleep(0.5)
                continue
            elif res.status_code != 200:
                retry_count += 1
                if retry_count > 3:
                    break
                time.sleep(0.2)
                continue

            data = res.json()
            if not data:
                break

            all_candles.extend(data)
            if len(data) < req_count:
                break

            to_param = data[-1]["candle_date_time_utc"]
            retry_count = 0
            time.sleep(0.08)  # 초당 요청 수 안전 유지를 위한 백오프 딜레이
        except Exception:
            time.sleep(0.2)
            continue

    return all_candles


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    markets = get_markets()

    all_price_patterns = []
    all_volume_patterns = []
    print(
        f"🔍 총 {len(markets)}개 종목 대상 5분 봉 대량 학습(가격 + 거래량 패턴) 시작..."
    )

    for idx, market in enumerate(markets):
        candles = fetch_5m_candles_deep(market, target_count=2000)
        if len(candles) < 100:
            continue

        df = pd.DataFrame(candles).sort_values("timestamp").reset_index(drop=True)

        for i in range(24, len(df) - 6):
            base_price = df.iloc[i]["trade_price"]
            future_max_price = df.iloc[i + 1 : i + 7]["trade_price"].max()

            if base_price > 0:
                surge_rate = (future_max_price - base_price) / base_price

                # 30분 내 8% 이상 상승 폭등 전조 구간 추출 (2시간 / 24개 봉)
                if surge_rate >= 0.08:
                    pre_prices = df.iloc[i - 24 : i]["trade_price"].values
                    pre_volumes = df.iloc[i - 24 : i][
                        "candle_acc_trade_volume"
                    ].values

                    # [개선 2] 거래량 스파이크(극단값) 왜곡 방지를 위한 로그 변환
                    log_volumes = np.log1p(pre_volumes)

                    p_min, p_max = pre_prices.min(), pre_prices.max()
                    v_min, v_max = log_volumes.min(), log_volumes.max()

                    if p_max > p_min and v_max > v_min:
                        # 가격 및 로그 거래량 정규화 (0~1)
                        norm_prices = (pre_prices - p_min) / (
                            p_max - p_min + 1e-8
                        )
                        norm_volumes = (log_volumes - v_min) / (
                            v_max - v_min + 1e-8
                        )

                        all_price_patterns.append(norm_prices)
                        all_volume_patterns.append(norm_volumes)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(markets):
            print(
                f"⌛ 진행률: {idx + 1}/{len(markets)} 코인 완료... (누적 급등 샘플 수: {len(all_price_patterns)}개)"
            )

    if all_price_patterns:
        # [개선 3] np.mean 대신 np.median(중앙값)을 사용하여 아웃라이어(이상치) 왜곡 차단
        golden_price_pattern = np.median(all_price_patterns, axis=0).tolist()
        golden_volume_pattern = np.median(all_volume_patterns, axis=0).tolist()

        pattern_data = {
            "golden_pattern": golden_price_pattern,
            "golden_volume_pattern": golden_volume_pattern,
            "sample_count": len(all_price_patterns),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(PATTERN_FILE, "w", encoding="utf-8") as f:
            json.dump(pattern_data, f, ensure_ascii=False, indent=4)

        print(
            f"\n✅ 총 {len(all_price_patterns)}개의 폭등 전조 샘플 학습 완료! 'golden_pattern.json' 저장 완료."
        )
    else:
        print("⚠️ 급등 패턴을 찾지 못했습니다.")


if __name__ == "__main__":
    main()
