import os
import json
import time
from datetime import datetime
import sqlite3
import requests
import pandas as pd
import streamlit as st

# 페이지 설정 (반드시 맨 처음에 위치해야 합니다)
st.set_page_config(
    page_title="업비트 AI 자가진화 실시간 대시보드",
    page_icon="🧬",
    layout="wide"
)

WEIGHTS_FILE = "strategy_weights.json"
DB_FILE = "upbit_history.db"

# 세션 상태(Session State)를 이용해 이전 순위 및 캐시 데이터 관리
if 'previous_ranks' not in st.session_state:
    st.session_state.previous_ranks = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'cached_results' not in st.session_state:
    st.session_state.cached_results = []
if 'generation' not in st.session_state:
    st.session_state.generation = 1

def init_db():
    """SQLite DB 및 테이블 초기화 함수"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analyzed_at TEXT,
            market TEXT,
            korean_name TEXT,
            trade_price REAL,
            signed_change_rate REAL,
            rank INTEGER,
            net_buy_plus_count_3h INTEGER,
            power_70_plus_count INTEGER,
            consecutive_bullish_count INTEGER,
            volume_spike_count INTEGER,
            iceberg_status TEXT,
            score REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_to_db(records):
    """분석 결과를 SQLite DB에 누적 저장"""
    if not records:
        return
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.DataFrame(records)
        df['analyzed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if '거래대금순위' in df.columns:
            df = df.rename(columns={'거래대금순위': 'rank'})
        if 'surge_rank_diff' in df.columns:
            df = df.drop(columns=['surge_rank_diff'])
            
        df.to_sql("analysis_history", conn, if_exists="append", index=False)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB 저장 오류: {e}")

def load_weights():
    default_weights = {
        "vol_up_weight": 2.0, "power_70_weight": 2.5, "net_buy_weight": 3.0,
        "consecutive_bullish_weight": 3.5, "volume_spike_weight": 4.0,
        "iceberg_accumulation_weight": 4.5, "rank_change_weight": 1.0,
        "cryptoquant_weight": 5.0, "new_listing_boost": 5.0,
        "volume_surge_weight": 6.0, "breakout_weight": 7.0, "generation": 1
    }
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                for k, v in default_weights.items():
                    if k not in loaded: loaded[k] = v
                return loaded
        except:
            return default_weights
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(default_weights, f, indent=4, ensure_ascii=False)
    return default_weights

def save_weights(weights):
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=4, ensure_ascii=False)

def evolve_weights(current_results):
    weights = load_weights()
    if not current_results:
        return weights

    top_coins = current_results[:5]
    avg_performance = sum(c['signed_change_rate'] for c in top_coins) / len(top_coins) if top_coins else 0
    adjustment_factor = 1.02 if avg_performance > 0 else 0.98
    
    weights["vol_up_weight"] = round(max(0.5, min(10.0, weights["vol_up_weight"] * adjustment_factor)), 2)
    weights["power_70_weight"] = round(max(0.5, min(10.0, weights["power_70_weight"] * adjustment_factor)), 2)
    weights["net_buy_weight"] = round(max(0.5, min(10.0, weights["net_buy_weight"] * adjustment_factor)), 2)
    weights["consecutive_bullish_weight"] = round(max(0.5, min(10.0, weights["consecutive_bullish_weight"] * adjustment_factor)), 2)
    weights["volume_spike_weight"] = round(max(0.5, min(10.0, weights["volume_spike_weight"] * adjustment_factor)), 2)
    
    weights["generation"] = weights.get("generation", 1) + 1
    save_weights(weights)
    return weights

class UpbitEvolutionAnalyzer:
    def __init__(self, weights):
        self.base_url = "https://api.upbit.com/v1"
        self.weights = weights

    def get_krw_markets(self):
        url = f"{self.base_url}/market/all"
        res = requests.get(url).json()
        return [item for item in res if item['market'].startswith('KRW-')]

    def get_ticker_info(self, markets):
        market_str = ",".join([m['market'] for m in markets])
        url = f"{self.base_url}/ticker?markets={market_str}"
        res = requests.get(url).json()
        return {item['market']: item for item in res}

    def get_candles(self, market, count=30):
        url = f"{self.base_url}/candles/minutes/15?market={market}&count={count}"
        res = requests.get(url).json()
        if not isinstance(res, list): return []
        res.reverse()
        return res

    def analyze_coin(self, market_info, ticker):
        market = market_info['market']
        korean_name = market_info['korean_name']
        trade_price = ticker['trade_price']
        signed_change_rate = ticker['signed_change_rate'] * 100

        candles = self.get_candles(market, count=30)
        if len(candles) < 1: return None

        max_high = max(c['high_price'] for c in candles)
        drop_from_high = ((max_high - trade_price) / max_high) * 100 if max_high > 0 else 0

        last_candle = candles[-1]
        is_rebounding = last_candle['trade_price'] > last_candle['opening_price']
        
        if drop_from_high >= 7.0 and not is_rebounding:
            return None

        vol_up_count, power_70_plus_count, volume_spike_count = 0, 0, 0
        acc_signals, dist_signals = 0, 0

        for i in range(1, len(candles)):
            prev_vol = candles[i-1]['candle_acc_trade_volume']
            curr_vol = candles[i]['candle_acc_trade_volume']
            if curr_vol > prev_vol: vol_up_count += 1
            if prev_vol > 0 and curr_vol >= (prev_vol * 2.0): volume_spike_count += 1

            high, low, close, open_p = candles[i]['high_price'], candles[i]['low_price'], candles[i]['trade_price'], candles[i]['opening_price']
            price_range_pct = ((high - low) / low) * 100 if low > 0 else 0
            if prev_vol > 0 and curr_vol >= (prev_vol * 2.5) and price_range_pct <= 1.5:
                if close >= open_p: acc_signals += 1
                else: dist_signals += 1

            volume_power = ((close - low) / (high - low)) * 100 if high != low else 50.0
            if volume_power >= 70.0: power_70_plus_count += 1

        recent_1h_candles = candles[-4:]
        consecutive_bullish_count = sum(1 for c in recent_1h_candles if c['trade_price'] > c['opening_price'])
        net_buy_plus_count_3h = sum(1 for c in candles[-12:] if c['trade_price'] - c['opening_price'] > 0)

        iceberg_status = "🟢 매집 우세" if acc_signals > dist_signals else ("🔴 매도 우세" if dist_signals > acc_signals else "⚪ 중립")
        
        score = (vol_up_count * self.weights['vol_up_weight']) + \
                (power_70_plus_count * self.weights['power_70_weight']) + \
                (net_buy_plus_count_3h * self.weights['net_buy_weight']) + \
                (consecutive_bullish_count * self.weights['consecutive_bullish_weight']) + \
                (volume_spike_count * self.weights['volume_spike_weight'])

        if signed_change_rate >= 20.0 and drop_from_high <= 5.0:
            score += 15.0

        return {
            'market': market, 'korean_name': korean_name, 'trade_price': trade_price,
            'signed_change_rate': round(signed_change_rate, 2),
            'net_buy_plus_count_3h': net_buy_plus_count_3h,
            'power_70_plus_count': power_70_plus_count,
            'vol_up_count': vol_up_count,
            'consecutive_bullish_count': consecutive_bullish_count,
            'volume_spike_count': volume_spike_count,
            'iceberg_status': iceberg_status,
            'score': round(score, 2)
        }

def run_analysis_process():
    weights = load_weights()
    analyzer = UpbitEvolutionAnalyzer(weights)
    krw_markets = analyzer.get_krw_markets()
    ticker_dict = analyzer.get_ticker_info(krw_markets)

    filtered_markets = [m for m in krw_markets if ticker_dict.get(m['market'], {}).get('acc_trade_price_24h', 0) >= 1_000_000]
    sorted_tickers = sorted(ticker_dict.items(), key=lambda x: x[1].get('acc_trade_price_24h', 0), reverse=True)
    current_ranks = {m: r + 1 for r, (m, _) in enumerate(sorted_tickers)}

    results = []
    for market_info in filtered_markets:
        m_code = market_info['market']
        ticker = ticker_dict.get(m_code)
        if not ticker: continue
        res = analyzer.analyze_coin(market_info, ticker)
        if not res: continue
        res['거래대금순위'] = current_ranks.get(m_code, 999)
        results.append(res)
        time.sleep(0.01)

    df = pd.DataFrame(results).sort_values(by='score', ascending=False).reset_index(drop=True)
    
    new_previous_ranks = {}
    for idx, row in df.iterrows():
        current_rank = idx + 1
        m_code = row['market']
        new_previous_ranks[m_code] = current_rank
        
        if m_code in st.session_state.previous_ranks:
            prev_rank = st.session_state.previous_ranks[m_code]
            rank_diff = prev_rank - current_rank
            df.at[idx, 'surge_rank_diff'] = int(rank_diff) if rank_diff >= 10 else 0
        else:
            df.at[idx, 'surge_rank_diff'] = 0

    st.session_state.previous_ranks = new_previous_ranks
    records = df.to_dict(orient='records')

    save_to_db(records)
    evolved_weights = evolve_weights(records)
    
    return records, evolved_weights['generation']

# UI 헤더 구성
st.markdown("<h1 style='text-align: center; color: #093687;'>🧬 업비트 AI 자가진화형 실시간 분석 대시보드</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    gen_placeholder = st.empty()
    time_placeholder = st.empty()

# 최초 실행 혹은 데이터가 없을 때 실행
if not st.session_state.cached_results:
    with st.spinner("🚀 AI가 전체 코인을 분석하고 진화 알고리즘을 적용 중입니다..."):
        results, generation = run_analysis_process()
        st.session_state.cached_results = results
        st.session_state.generation = generation
        st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

gen_placeholder.markdown(f"<div style='text-align: center;'><span style='background: #093687; color: white; padding: 4px 12px; border-radius: 12px; font-size: 14px;'>🧬 현재 진화 세대 (Generation): {st.session_state.generation}</span></div>", unsafe_allow_html=True)
time_placeholder.markdown(f"<p style='text-align: center; color: #666; margin-top: 8px;'>마지막 업데이트: {st.session_state.last_update} (총 {len(st.session_state.cached_results)}개 코인 분석됨)</p>", unsafe_allow_html=True)

# 검색 바 기능
search_query = st.text_input("🔍 코인명 검색 (예: 비트코인, BTC)", "").strip().lower()

# 데이터프레임 가공
display_data = []
for i, row in enumerate(st.session_state.cached_results):
    surge_text = f" 🚀 +{row['surge_rank_diff']}위 급등" if row.get('surge_rank_diff', 0) >= 10 else ""
    display_data.append({
        "순위": i + 1,
        "코인명": f"{row['korean_name']} ({row['market'].replace('KRW-', '')}){surge_text}",
        "현재가격": f"{row['trade_price']:,} 원",
        "등락율": f"{row['signed_change_rate']}%",
        "거래대금순위": f"{row['거래대금순위']}위",
        "3시간순매수": f"{row['net_buy_plus_count_3h']}회",
        "체결강도70%+": f"{row['power_70_plus_count']}회",
        "연속양봉": f"{row['consecutive_bullish_count']}회",
        "2배폭증": f"{row['volume_spike_count']}회",
        "아이스버그": row['iceberg_status'],
        "최종점수": row['score'],
        "raw_name": f"{row['korean_name']} {row['market']}".lower()
    })

df_display = pd.DataFrame(display_data)

# 검색 필터링 적용
if search_query:
    df_filtered = df_display[df_display['raw_name'].str.contains(search_query)]
else:
    df_filtered = df_display

# raw_name 컬럼은 화면에 보이지 않도록 제거
df_show = df_filtered.drop(columns=['raw_name'])

# 테이블 출력
st.dataframe(df_show, use_container_width=True, hide_index=True)

# 자동 새로고침 버튼 (또는 수동 갱신)
if st.button("🔄 데이터 새로고침 및 재분석"):
    with st.spinner("AI가 최신 데이터를 다시 분석하는 중입니다..."):
        results, generation = run_analysis_process()
        st.session_state.cached_results = results
        st.session_state.generation = generation
        st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()