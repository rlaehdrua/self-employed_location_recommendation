import streamlit as st
import pandas as pd
import numpy as np
import os
import unicodedata
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="소상공인 창업 입지 추천",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 3rem 2rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 2.5rem; font-weight: 700; margin: 0 0 0.5rem 0; }
.hero p  { font-size: 1.1rem; opacity: 0.9; margin: 0; }

.result-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 50%;
    width: 38px;
    height: 38px;
    font-weight: 700;
    font-size: 1rem;
    margin-right: 0.6rem;
    flex-shrink: 0;
}
.score-bar {
    background: #f0f0f0;
    border-radius: 8px;
    height: 10px;
    margin-top: 6px;
    overflow: hidden;
}
.score-fill {
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 8px;
    height: 10px;
}
.metric-pill {
    display: inline-block;
    background: #f0f4ff;
    color: #4a5568;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    margin: 2px;
}
.reason-chip {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.82rem;
    margin: 2px;
}
.warning-chip {
    display: inline-block;
    background: #fff3e0;
    color: #e65100;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.82rem;
    margin: 2px;
}
.rent-box {
    background: #f8faff;
    border: 1px solid #dce6ff;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-top: 0.6rem;
    font-size: 0.9rem;
}
.achieve-box {
    background: #f8f8f8;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    def find_files(keyword1, keyword2=''):
        results = []
        for f in sorted(os.listdir('.')):
            nfc = unicodedata.normalize('NFC', f)
            if keyword1 in nfc and (not keyword2 or keyword2 in nfc) and f.endswith('.csv'):
                results.append(f)
        return results

    RENAME_2025 = {
        'stdr_yyqu_cd': '기준_년분기_코드',
        'trdar_se_cd': '상권_구분_코드',
        'trdar_se_cd_nm': '상권_구분_코드_명',
        'trdar_cd': '상권_코드',
        'trdar_cd_nm': '상권_코드_명',
        'svc_induty_cd': '서비스_업종_코드',
        'svc_induty_cd_nm': '서비스_업종_코드_명',
        'stor_co': '점포_수',
        'similr_induty_stor_co': '유사_업종_점포_수',
        'opbiz_rt': '개업_율',
        'opbiz_stor_co': '개업_점포_수',
        'clsbiz_rt': '폐업_률',
        'clsbiz_stor_co': '폐업_점포_수',
        'frc_stor_co': '프랜차이즈_점포_수',
    }

    store_frames, sales_frames = [], []

    store_files = find_files('점포', '상권')
    sales_files = find_files('추정매출', '상권')

    for f in store_files:
        try:
            df = pd.read_csv(f, encoding='cp949', low_memory=False)
            if df.columns[0] != '기준_년분기_코드':
                df.rename(columns=RENAME_2025, inplace=True)
            store_frames.append(df)
        except Exception:
            pass

    for f in sales_files:
        try:
            df = pd.read_csv(f, encoding='cp949', low_memory=False)
            sales_frames.append(df)
        except Exception:
            pass

    if not store_frames or not sales_frames:
        return None, None, None

    store_df = pd.concat(store_frames, ignore_index=True)
    sales_df = pd.concat(sales_frames, ignore_index=True)

    # ── 매출 컬럼 탐지 ──────────────────────────────────────────────────────
    total_sales_col = None
    for cand in ['당월_매출_금액', '분기_매출_금액']:
        if cand in sales_df.columns:
            total_sales_col = cand
            break
    if total_sales_col is None:
        cands = [c for c in sales_df.columns if '매출_금액' in c or '매출금액' in c]
        total_sales_col = cands[0] if cands else None

    age_2030_cols = [c for c in sales_df.columns if ('20대' in c or '30대' in c) and '매출' in c]
    male_cols     = [c for c in sales_df.columns if '남성' in c and '매출' in c]
    female_cols   = [c for c in sales_df.columns if '여성' in c and '매출' in c]
    weekday_cols  = [c for c in sales_df.columns if '주중' in c and '매출' in c]
    weekend_cols  = [c for c in sales_df.columns if '주말' in c and '매출' in c]

    agg_dict = {}
    if total_sales_col:
        agg_dict[total_sales_col] = 'sum'
    for col in age_2030_cols + male_cols + female_cols + weekday_cols + weekend_cols:
        if col in sales_df.columns:
            agg_dict[col] = 'sum'

    key_cols = ['기준_년분기_코드', '상권_코드', '서비스_업종_코드']
    sales_agg = sales_df.groupby(key_cols).agg(agg_dict).reset_index()

    merge_df = pd.merge(store_df, sales_agg, on=key_cols, how='inner')

    # ── 파생 지표 계산 ───────────────────────────────────────────────────────
    if total_sales_col and total_sales_col in merge_df.columns:
        merge_df['점포당매출'] = (
            merge_df[total_sales_col] / merge_df['점포_수'].replace(0, np.nan)
        )
    else:
        merge_df['점포당매출'] = np.nan

    if age_2030_cols:
        exists_2030 = [c for c in age_2030_cols if c in merge_df.columns]
        merge_df['매출_2030'] = merge_df[exists_2030].sum(axis=1)
        if total_sales_col in merge_df.columns:
            denom = merge_df[total_sales_col].replace(0, np.nan)
            merge_df['비중_2030'] = merge_df['매출_2030'] / denom
        else:
            merge_df['비중_2030'] = np.nan
    else:
        merge_df['비중_2030'] = np.nan

    if male_cols and female_cols:
        m_exist = [c for c in male_cols   if c in merge_df.columns]
        f_exist = [c for c in female_cols if c in merge_df.columns]
        m = merge_df[m_exist].sum(axis=1)
        f = merge_df[f_exist].sum(axis=1)
        total_mf = (m + f).replace(0, np.nan)
        merge_df['남성비중'] = m / total_mf
        merge_df['여성비중'] = f / total_mf
    else:
        merge_df['남성비중'] = 0.5
        merge_df['여성비중'] = 0.5

    if weekday_cols and weekend_cols:
        wd_exist = [c for c in weekday_cols if c in merge_df.columns]
        we_exist = [c for c in weekend_cols if c in merge_df.columns]
        wd = merge_df[wd_exist].sum(axis=1)
        we = merge_df[we_exist].sum(axis=1)
        total_ww = (wd + we).replace(0, np.nan)
        merge_df['주중비중'] = wd / total_ww
        merge_df['주말비중'] = we / total_ww
    else:
        merge_df['주중비중'] = 0.5
        merge_df['주말비중'] = 0.5

    # ── 최신 분기 기준 요약 ──────────────────────────────────────────────────
    latest_q = merge_df['기준_년분기_코드'].max()
    latest = merge_df[merge_df['기준_년분기_코드'] == latest_q].copy()

    group_cols = ['상권_코드', '상권_코드_명', '상권_구분_코드_명', '서비스_업종_코드_명']
    agg_s = {
        '점포_수':          ('점포_수', 'sum'),
        '프랜차이즈_점포_수': ('프랜차이즈_점포_수', 'sum'),
        '점포당매출':        ('점포당매출', 'mean'),
        '비중_2030':         ('비중_2030', 'mean'),
        '남성비중':          ('남성비중', 'mean'),
        '여성비중':          ('여성비중', 'mean'),
        '주중비중':          ('주중비중', 'mean'),
        '주말비중':          ('주말비중', 'mean'),
        '개업_율':           ('개업_율', 'mean'),
        '폐업_률':           ('폐업_률', 'mean'),
    }
    summary = latest.groupby(group_cols).agg(**agg_s).reset_index()
    summary.rename(columns={
        '점포_수':          '총점포수',
        '프랜차이즈_점포_수': '프랜차이즈수',
        '개업_율':          '개업률',
        '폐업_률':          '폐업률',
    }, inplace=True)

    mx = summary['총점포수'].max()
    summary['경쟁강도'] = summary['총점포수'] / mx if mx > 0 else 0.0

    # ── 임대료 데이터 ────────────────────────────────────────────────────────
    rent_df = None
    rent_files = find_files('임대', '소규모')
    for f in rent_files:
        try:
            rdf = pd.read_csv(f, encoding='cp949', header=None, skiprows=2)
            expected = ['No', '지역', '권역', '세부지역',
                        '2024Q3', '2024Q4', '2025Q1', '2025Q2',
                        '2025Q3', '2025Q4', '2026Q1']
            rdf.columns = expected[:len(rdf.columns)]
            rdf = rdf.dropna(subset=['지역']).reset_index(drop=True)
            for c in ['2024Q3', '2024Q4', '2025Q1', '2025Q2',
                      '2025Q3', '2025Q4', '2026Q1']:
                if c in rdf.columns:
                    rdf[c] = pd.to_numeric(rdf[c], errors='coerce')
            latest_rent_col = None
            for col in ['2026Q1', '2025Q4', '2025Q3']:
                if col in rdf.columns:
                    latest_rent_col = col
                    break
            if latest_rent_col:
                rdf['최신임대료'] = rdf[latest_rent_col]
                for col in ['2025Q4', '2025Q3', '2025Q2']:
                    if col in rdf.columns:
                        rdf['최신임대료'] = rdf['최신임대료'].fillna(rdf[col])
            rent_df = rdf
            break
        except Exception:
            pass

    industry_list = sorted(latest['서비스_업종_코드_명'].dropna().unique().tolist())
    return summary, rent_df, industry_list


# ── Helper Functions ──────────────────────────────────────────────────────────
def safe_scale(series):
    arr = series.fillna(0).values.reshape(-1, 1)
    mn, mx = float(arr.min()), float(arr.max())
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return pd.Series(
        MinMaxScaler().fit_transform(arr).flatten(),
        index=series.index
    )


def recommend(df, industry, age_targets, gender, schedule):
    sub = df[df['서비스_업종_코드_명'] == industry].copy()
    if sub.empty:
        return pd.DataFrame()

    # 연령 점수 (30%)
    if '2030' in age_targets or '20대' in age_targets or '30대' in age_targets:
        age_raw = sub['비중_2030'].fillna(0)
    else:
        age_raw = 1 - sub['비중_2030'].fillna(0)   # 40대 이상 선호 → 2030 비중 낮을수록 좋음
    age_score = safe_scale(age_raw)

    # 업종 수요 점수 (25%) — 점포당 매출 기반
    demand_score = safe_scale(sub['점포당매출'].fillna(0))

    # 시간/요일 점수 (20%)
    if schedule == '주중':
        time_score = safe_scale(sub['주중비중'].fillna(0.5))
    elif schedule == '주말':
        time_score = safe_scale(sub['주말비중'].fillna(0.5))
    else:
        time_score = pd.Series(0.5, index=sub.index)

    # 성별 점수 (10%)
    if gender == '남성':
        gender_score = safe_scale(sub['남성비중'].fillna(0.5))
    elif gender == '여성':
        gender_score = safe_scale(sub['여성비중'].fillna(0.5))
    else:
        gender_score = pd.Series(0.5, index=sub.index)

    # 경쟁 효율 점수 (15%) — 경쟁강도 낮을수록 고점
    comp_score = safe_scale(1 - sub['경쟁강도'].fillna(0.5))

    sub = sub.copy()
    sub['총점']   = (0.30 * age_score.values
                   + 0.25 * demand_score.values
                   + 0.20 * time_score.values
                   + 0.10 * gender_score.values
                   + 0.15 * comp_score.values)
    sub['연령점수'] = age_score.values
    sub['수요점수'] = demand_score.values
    sub['패턴점수'] = time_score.values
    sub['성별점수'] = gender_score.values
    sub['경쟁점수'] = comp_score.values

    return sub.nlargest(5, '총점').reset_index(drop=True)


def get_rent_for_district(rent_df, district_name):
    if rent_df is None or district_name is None:
        return None
    district_name = str(district_name)
    # 세부지역 매칭 (가장 정확)
    for _, row in rent_df.iterrows():
        loc = unicodedata.normalize('NFC', str(row.get('세부지역', '') or ''))
        for token in loc.split():
            if len(token) >= 2 and token in district_name:
                val = row.get('최신임대료')
                if pd.notna(val):
                    return float(val)
    # 지역 매칭 (폴백)
    for _, row in rent_df.iterrows():
        loc = unicodedata.normalize('NFC', str(row.get('지역', '') or ''))
        for token in loc.split():
            if len(token) >= 2 and token in district_name:
                val = row.get('최신임대료')
                if pd.notna(val):
                    return float(val)
    return None


def calc_rent(val_per_sqm, pyeong):
    """val_per_sqm: 천원/㎡ (월), pyeong: 평 → (월임대료원, 연임대료원)"""
    sqm = pyeong * 3.306
    monthly = val_per_sqm * 1000 * sqm
    annual  = monthly * 12
    return int(monthly), int(annual)


def build_reasons(row, age_targets, gender, schedule, desired_monthly):
    reasons  = []
    warnings = []

    # 2030 비중
    p2030 = row.get('비중_2030', 0) or 0
    if '2030' in age_targets or '20대' in age_targets or '30대' in age_targets:
        if p2030 >= 0.5:
            reasons.append(f"2030 매출 비중 {p2030*100:.0f}%로 높음")
        else:
            warnings.append(f"2030 고객 비중 {p2030*100:.0f}%로 낮음")
    else:
        if p2030 <= 0.4:
            reasons.append(f"중장년 고객 비중 높음 ({(1-p2030)*100:.0f}%)")

    # 점포당 매출
    pm = row.get('점포당매출', 0) or 0
    if pm > 0:
        if pm >= desired_monthly:
            reasons.append(f"평균 점포당 매출 {pm/1e4:.0f}만원으로 목표 달성 가능")
        elif pm >= desired_monthly * 0.7:
            reasons.append(f"평균 점포당 매출 {pm/1e4:.0f}만원 (목표의 {pm/desired_monthly*100:.0f}%)")
        else:
            warnings.append(f"점포당 매출 {pm/1e4:.0f}만원으로 목표 대비 낮음")

    # 경쟁강도
    comp = row.get('경쟁강도', 0.5)
    if comp <= 0.25:
        reasons.append("경쟁 밀도 낮은 블루오션 상권")
    elif comp >= 0.75:
        warnings.append("경쟁이 치열한 레드오션 상권")

    # 시간 패턴
    if schedule == '주중' and row.get('주중비중', 0.5) >= 0.6:
        reasons.append(f"주중 매출 집중 상권 ({row['주중비중']*100:.0f}%)")
    elif schedule == '주말' and row.get('주말비중', 0.5) >= 0.5:
        reasons.append(f"주말 매출 활성 상권 ({row['주말비중']*100:.0f}%)")

    # 성별
    if gender == '여성' and (row.get('여성비중', 0.5) or 0.5) >= 0.55:
        reasons.append(f"여성 고객 비중 {row['여성비중']*100:.0f}%")
    elif gender == '남성' and (row.get('남성비중', 0.5) or 0.5) >= 0.55:
        reasons.append(f"남성 고객 비중 {row['남성비중']*100:.0f}%")

    return reasons, warnings


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Hero
    st.markdown("""
<div class="hero">
  <h1>🏪 소상공인 창업 입지 추천</h1>
  <p>서울시 상권 빅데이터 기반 · 당신에게 맞는 창업 입지 TOP 5를 추천해드립니다</p>
</div>
""", unsafe_allow_html=True)

    # 데이터 로딩
    with st.spinner("데이터를 불러오는 중..."):
        summary, rent_df, industry_list = load_data()

    if summary is None:
        st.error("CSV 데이터 파일을 찾을 수 없습니다. app.py와 같은 폴더에 CSV 파일이 있는지 확인해주세요.")
        st.stop()

    # 요약 통계
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 상권 수",  f"{summary['상권_코드'].nunique():,}개")
    c2.metric("분석 업종 수",  f"{summary['서비스_업종_코드_명'].nunique():,}개")
    c3.metric("분석 레코드",   f"{len(summary):,}건")
    c4.metric("임대료 지역",   f"{len(rent_df):,}개" if rent_df is not None else "N/A")

    st.markdown("---")

    # ── 입력 폼 ──────────────────────────────────────────────────────────────
    st.markdown("## 📝 창업 정보 입력")

    with st.form("rec_form"):
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("### 💰 자본금 & 업종")
            capital = st.number_input(
                "투자 가능 자본금 (만원)",
                min_value=500, max_value=100000, value=5000, step=500,
                help="보증금 + 인테리어 + 초기 운영비 등 총 투자 가능 금액"
            )
            pyeong = st.slider(
                "희망 가게 크기 (평)",
                min_value=5, max_value=50, value=10, step=5,
                help="10평 = 33.06㎡ 기준으로 임대료를 추산합니다"
            )
            industry = st.selectbox(
                "희망 업종",
                options=industry_list if industry_list else ["데이터 없음"]
            )

        with col_r:
            st.markdown("### 👥 타겟 고객층")
            age_targets = st.multiselect(
                "타겟 연령대 (복수 선택 가능)",
                options=["2030", "40대", "50대 이상"],
                default=["2030"]
            )
            gender = st.radio(
                "타겟 성별",
                options=["무관", "남성", "여성"],
                horizontal=True
            )
            schedule = st.radio(
                "주요 운영 패턴",
                options=["무관", "주중", "주말"],
                horizontal=True,
                help="주로 어느 시기에 매출이 발생하길 원하시나요?"
            )
            desired_sales = st.number_input(
                "희망 월 매출 (만원)",
                min_value=100, max_value=100000, value=2000, step=100
            )

        submitted = st.form_submit_button(
            "🔍  창업 입지 추천받기",
            use_container_width=True,
            type="primary"
        )

    # ── 결과 ─────────────────────────────────────────────────────────────────
    if submitted:
        if not age_targets:
            age_targets = ["무관"]

        desired_monthly = desired_sales * 10000

        with st.spinner("최적 입지 분석 중..."):
            results = recommend(summary, industry, age_targets, gender, schedule)

        if results.empty:
            st.warning(f"'{industry}' 업종 데이터가 충분하지 않습니다. 다른 업종을 선택해보세요.")
            st.stop()

        st.markdown("---")
        st.markdown(f"## 🏆 추천 창업 입지 TOP 5")
        st.caption(
            f"업종: **{industry}** | 자본금: **{capital:,}만원** | 가게크기: **{pyeong}평** | "
            f"타겟: **{', '.join(age_targets)}** · **{gender}** | 운영패턴: **{schedule}** | "
            f"희망매출: **{desired_sales:,}만원/월**"
        )

        for rank_i, row in results.iterrows():
            rank = rank_i + 1
            score_pct = int(row['총점'] * 100)
            pm = float(row.get('점포당매출') or 0)
            p2030  = float(row.get('비중_2030') or 0)
            comp   = float(row.get('경쟁강도') or 0)
            opbiz  = float(row.get('개업률') or 0)
            clbiz  = float(row.get('폐업률') or 0)
            n_stor = int(row.get('총점포수') or 0)

            reasons, warnings_list = build_reasons(row, age_targets, gender, schedule, desired_monthly)

            # 임대료
            rent_val = get_rent_for_district(rent_df, row['상권_코드_명'])
            if rent_val:
                monthly_rent, annual_rent = calc_rent(rent_val, pyeong)
                rent_ratio = monthly_rent / (capital * 10000) * 100
                rent_html = (
                    f'<div class="rent-box">'
                    f'<strong>💰 임대료 추산 ({pyeong}평 기준)</strong> &nbsp;—&nbsp; '
                    f'월 <strong>{monthly_rent:,.0f}원</strong> &nbsp;|&nbsp; '
                    f'연 <strong>{annual_rent:,.0f}원</strong> &nbsp;|&nbsp; '
                    f'자본금 대비 <strong>{rent_ratio:.1f}%</strong>'
                    f'</div>'
                )
            else:
                rent_html = (
                    '<div class="rent-box" style="color:#999;">'
                    '💰 임대료 데이터 매칭 없음 (지역명이 다를 수 있음)'
                    '</div>'
                )

            # 희망매출 달성 가능성
            if pm > 0:
                ratio = pm / desired_monthly * 100
                if ratio >= 100:
                    ac_color = '#2e7d32'
                    ac_label = f'✅ 달성 가능 ({ratio:.0f}%)'
                elif ratio >= 70:
                    ac_color = '#f57f17'
                    ac_label = f'⚠️ 달성 도전 ({ratio:.0f}%)'
                else:
                    ac_color = '#c62828'
                    ac_label = f'❌ 달성 어려움 ({ratio:.0f}%)'
                achieve_html = (
                    f'<div class="achieve-box">'
                    f'<strong>📈 희망매출 달성 가능성:</strong> '
                    f'<span style="color:{ac_color}; font-weight:600;">{ac_label}</span>'
                    f'&nbsp; (평균 점포당 매출 <strong>{pm/1e4:.0f}만원</strong> vs '
                    f'목표 <strong>{desired_sales:,}만원</strong>)'
                    f'</div>'
                )
            else:
                achieve_html = (
                    '<div class="achieve-box" style="color:#999;">'
                    '📈 매출 데이터 없음'
                    '</div>'
                )

            # 핵심 지표 pills
            pills_html = (
                f'<span class="metric-pill">총점포 {n_stor}개</span>'
                f'<span class="metric-pill">점포당매출 {pm/1e4:.0f}만원</span>'
                f'<span class="metric-pill">2030비중 {p2030*100:.0f}%</span>'
                f'<span class="metric-pill">경쟁강도 {comp*100:.0f}%</span>'
                f'<span class="metric-pill">개업률 {opbiz:.1f}%</span>'
                f'<span class="metric-pill">폐업률 {clbiz:.1f}%</span>'
            )

            reason_html  = "".join(f'<span class="reason-chip">✓ {r}</span>' for r in reasons)
            warning_html = "".join(f'<span class="warning-chip">⚠ {w}</span>' for w in warnings_list)

            card = f"""
<div class="result-card">
  <div style="display:flex; align-items:center; margin-bottom:0.75rem;">
    <span class="rank-badge">{rank}</span>
    <div style="flex:1;">
      <div style="font-size:1.2rem; font-weight:700; color:#2d3748;">{row['상권_코드_명']}</div>
      <div style="font-size:0.85rem; color:#718096;">{row.get('상권_구분_코드_명', '')}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:1.6rem; font-weight:700; color:#667eea;">{score_pct}점</div>
      <div style="font-size:0.75rem; color:#718096;">적합도</div>
    </div>
  </div>
  <div class="score-bar"><div class="score-fill" style="width:{score_pct}%;"></div></div>
  <div style="margin-top:0.75rem;">{pills_html}</div>
  {rent_html}
  {achieve_html}
  <div style="margin-top:0.75rem;">
    <strong>💡 추천 사유</strong><br>
    {reason_html}{warning_html}
  </div>
</div>
"""
            st.markdown(card, unsafe_allow_html=True)

        # ── 비교 테이블 ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 추천 상권 비교표")

        disp = results[[
            '상권_코드_명', '상권_구분_코드_명', '총점포수',
            '점포당매출', '비중_2030', '경쟁강도', '개업률', '폐업률', '총점'
        ]].copy()
        disp.rename(columns={
            '상권_코드_명':     '상권명',
            '상권_구분_코드_명': '상권유형',
            '총점포수':         '총점포수',
            '점포당매출':       '점포당매출(만원)',
            '비중_2030':        '2030비중(%)',
            '경쟁강도':         '경쟁강도(%)',
            '개업률':           '개업률(%)',
            '폐업률':           '폐업률(%)',
            '총점':             '적합도점수',
        }, inplace=True)
        disp['점포당매출(만원)'] = (disp['점포당매출(만원)'] / 1e4).round(0).astype('Int64')
        disp['2030비중(%)']    = (disp['2030비중(%)'] * 100).round(1)
        disp['경쟁강도(%)']    = (disp['경쟁강도(%)'] * 100).round(1)
        disp['적합도점수']     = (disp['적합도점수'] * 100).round(1)
        disp.index = range(1, len(disp) + 1)
        st.dataframe(disp, use_container_width=True)

        # ── 점수 구성 ─────────────────────────────────────────────────────────
        st.markdown("### 🎯 적합도 점수 구성 (0~1)")
        score_disp = results[['상권_코드_명', '연령점수', '수요점수', '패턴점수', '성별점수', '경쟁점수', '총점']].copy()
        score_disp.rename(columns={
            '상권_코드_명': '상권명',
            '연령점수':    '연령(30%)',
            '수요점수':    '업종수요(25%)',
            '패턴점수':    '시간패턴(20%)',
            '성별점수':    '성별(10%)',
            '경쟁점수':    '경쟁효율(15%)',
            '총점':        '종합점수',
        }, inplace=True)
        for c in score_disp.columns[1:]:
            score_disp[c] = score_disp[c].round(3)
        score_disp.index = range(1, len(score_disp) + 1)
        st.dataframe(score_disp, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "📊 데이터 출처: 서울시 열린데이터광장 상권분석서비스 (2021~2025) · 한국감정원 소규모 상가 임대동향 | "
        "⚖️ 가중치: 연령 30% · 업종수요 25% · 시간패턴 20% · 성별 10% · 경쟁효율 15%"
    )


if __name__ == "__main__":
    main()
