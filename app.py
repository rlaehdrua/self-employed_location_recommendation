import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
import gdown

st.set_page_config(
    page_title="소상공인 창업 입지 추천",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Google Drive 파일 ID ──────────────────────────────────────────────────────
GDRIVE_FILES = {
    '점포_상권_2025.csv':    '1X_DqxOchOas6wrU_HVTEbor6FvVLH4YG',
    '추정매출_상권_2025.csv': '1eDMWhH3ibsOWCdekMzgnIJ_HKV9XWqnl',
    '영역_상권.csv':          '1_zlKlmhK_TtZH8yF3yOPSjlAXnN8YROk',
    '임대료_소규모상가.csv':   '1xRDJx8aXjH6-hEVbnYegyLXQTStiBEyF',
}
DATA_DIR = '/tmp/startup_rec_data'


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


# ── Google Drive 다운로드 ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def download_files():
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = {}
    for fname, fid in GDRIVE_FILES.items():
        out = os.path.join(DATA_DIR, fname)
        if not os.path.exists(out):
            gdown.download(id=fid, output=out, quiet=True, fuzzy=True)
        paths[fname] = out
    return paths


# ── 파일 읽기 (인코딩 자동 감지) ──────────────────────────────────────────────
def read_csv_safe(path, **kwargs):
    for enc in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"파일을 읽을 수 없습니다: {path}")


# ── 데이터 로딩 & 전처리 ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    paths = download_files()

    # ── 2025 컬럼명 영어→한국어 변환 ─────────────────────────────────────────
    RENAME_2025 = {
        'stdr_yyqu_cd':      '기준_년분기_코드',
        'trdar_se_cd':       '상권_구분_코드',
        'trdar_se_cd_nm':    '상권_구분_코드_명',
        'trdar_cd':          '상권_코드',
        'trdar_cd_nm':       '상권_코드_명',
        'svc_induty_cd':     '서비스_업종_코드',
        'svc_induty_cd_nm':  '서비스_업종_코드_명',
        'stor_co':           '점포_수',
        'similr_induty_stor_co': '유사_업종_점포_수',
        'opbiz_rt':          '개업_율',
        'opbiz_stor_co':     '개업_점포_수',
        'clsbiz_rt':         '폐업_률',
        'clsbiz_stor_co':    '폐업_점포_수',
        'frc_stor_co':       '프랜차이즈_점포_수',
    }

    # ── 점포 데이터 ───────────────────────────────────────────────────────────
    store_df = read_csv_safe(paths['점포_상권_2025.csv'], low_memory=False)
    if store_df.columns[0] != '기준_년분기_코드':
        store_df.rename(columns=RENAME_2025, inplace=True)

    # ── 추정매출 데이터 ───────────────────────────────────────────────────────
    sales_df = read_csv_safe(paths['추정매출_상권_2025.csv'], low_memory=False)

    # ── 영역_상권 데이터 (자치구·행정동 등 지역 정보) ─────────────────────────
    area_df = None
    try:
        area_raw = read_csv_safe(paths['영역_상권.csv'], low_memory=False)
        # 상권_코드 기준 최신 분기 1건만 유지
        q_col = [c for c in area_raw.columns if '년분기' in c or 'stdr_yyqu' in c]
        if q_col:
            latest_q = area_raw[q_col[0]].max()
            area_raw = area_raw[area_raw[q_col[0]] == latest_q]
        # 자치구·행정동 컬럼 탐지
        gu_col  = next((c for c in area_raw.columns if '자치구' in c and '명' in c), None)
        dong_col = next((c for c in area_raw.columns if '행정동' in c and '명' in c), None)
        cd_col   = next((c for c in area_raw.columns
                         if ('trdar_cd' == c or '상권_코드' == c) and '명' not in c), None)
        keep = [c for c in [cd_col, gu_col, dong_col] if c]
        if cd_col and keep:
            area_df = area_raw[keep].drop_duplicates(subset=[cd_col])
            area_df.rename(columns={cd_col: '상권_코드',
                                     gu_col:  '자치구명' if gu_col else None,
                                     dong_col: '행정동명' if dong_col else None},
                           inplace=True)
            area_df = area_df.dropna(axis=1, how='all')
    except Exception:
        area_df = None

    # ── 매출 컬럼 탐지 ────────────────────────────────────────────────────────
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
    sales_agg = sales_df.groupby(key_cols).agg(agg_dict).reset_index() if agg_dict else sales_df

    # ── 병합 ─────────────────────────────────────────────────────────────────
    merge_df = pd.merge(store_df, sales_agg, on=key_cols, how='inner')

    # ── 파생 지표 ─────────────────────────────────────────────────────────────
    if total_sales_col and total_sales_col in merge_df.columns:
        merge_df['점포당매출'] = (
            merge_df[total_sales_col] / merge_df['점포_수'].replace(0, np.nan)
        )
    else:
        merge_df['점포당매출'] = np.nan

    if age_2030_cols:
        exists_cols = [c for c in age_2030_cols if c in merge_df.columns]
        merge_df['매출_2030'] = merge_df[exists_cols].sum(axis=1)
        denom = merge_df[total_sales_col].replace(0, np.nan) if total_sales_col in merge_df.columns else np.nan
        merge_df['비중_2030'] = merge_df['매출_2030'] / denom
    else:
        merge_df['비중_2030'] = np.nan

    if male_cols and female_cols:
        m = merge_df[[c for c in male_cols   if c in merge_df.columns]].sum(axis=1)
        f = merge_df[[c for c in female_cols if c in merge_df.columns]].sum(axis=1)
        total_mf = (m + f).replace(0, np.nan)
        merge_df['남성비중'] = m / total_mf
        merge_df['여성비중'] = f / total_mf
    else:
        merge_df['남성비중'] = 0.5
        merge_df['여성비중'] = 0.5

    if weekday_cols and weekend_cols:
        wd = merge_df[[c for c in weekday_cols if c in merge_df.columns]].sum(axis=1)
        we = merge_df[[c for c in weekend_cols if c in merge_df.columns]].sum(axis=1)
        total_ww = (wd + we).replace(0, np.nan)
        merge_df['주중비중'] = wd / total_ww
        merge_df['주말비중'] = we / total_ww
    else:
        merge_df['주중비중'] = 0.5
        merge_df['주말비중'] = 0.5

    # ── 최신 분기 기준 상권×업종 요약 ────────────────────────────────────────
    latest_q = merge_df['기준_년분기_코드'].max()
    latest   = merge_df[merge_df['기준_년분기_코드'] == latest_q].copy()

    group_cols = ['상권_코드', '상권_코드_명', '상권_구분_코드_명', '서비스_업종_코드_명']
    summary = latest.groupby(group_cols).agg(
        총점포수      = ('점포_수',            'sum'),
        프랜차이즈수  = ('프랜차이즈_점포_수',  'sum'),
        점포당매출    = ('점포당매출',          'mean'),
        비중_2030     = ('비중_2030',           'mean'),
        남성비중      = ('남성비중',            'mean'),
        여성비중      = ('여성비중',            'mean'),
        주중비중      = ('주중비중',            'mean'),
        주말비중      = ('주말비중',            'mean'),
        개업률        = ('개업_율',             'mean'),
        폐업률        = ('폐업_률',             'mean'),
    ).reset_index()

    mx = summary['총점포수'].max()
    summary['경쟁강도'] = summary['총점포수'] / mx if mx > 0 else 0.0

    # ── 영역 정보 join (자치구명, 행정동명) ──────────────────────────────────
    if area_df is not None and '상권_코드' in area_df.columns:
        summary = summary.merge(area_df, on='상권_코드', how='left')

    # ── 임대료 데이터 ─────────────────────────────────────────────────────────
    rent_df = None
    try:
        rdf = read_csv_safe(paths['임대료_소규모상가.csv'], header=None, skiprows=2)
        expected = ['No', '지역', '권역', '세부지역',
                    '2024Q3', '2024Q4', '2025Q1', '2025Q2',
                    '2025Q3', '2025Q4', '2026Q1']
        rdf.columns = expected[:len(rdf.columns)]
        rdf = rdf.dropna(subset=['지역']).reset_index(drop=True)
        for c in ['2024Q3', '2024Q4', '2025Q1', '2025Q2',
                  '2025Q3', '2025Q4', '2026Q1']:
            if c in rdf.columns:
                rdf[c] = pd.to_numeric(rdf[c], errors='coerce')
        # 최신 임대료: 2026Q1 → 2025Q4 → 2025Q3 순으로 폴백
        rdf['최신임대료'] = np.nan
        for col in ['2026Q1', '2025Q4', '2025Q3', '2025Q2']:
            if col in rdf.columns:
                rdf['최신임대료'] = rdf['최신임대료'].fillna(rdf[col])
        rent_df = rdf
    except Exception:
        rent_df = None

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
        age_raw = 1 - sub['비중_2030'].fillna(0)
    age_score = safe_scale(age_raw)

    # 업종 수요 점수 (25%)
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

    # 경쟁 효율 점수 (15%)
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
    if rent_df is None or not district_name:
        return None
    district_name = str(district_name)
    for col in ['세부지역', '지역']:
        for _, row in rent_df.iterrows():
            loc = str(row.get(col, '') or '')
            for token in loc.split():
                if len(token) >= 2 and token in district_name:
                    val = row.get('최신임대료')
                    if pd.notna(val):
                        return float(val)
    return None


def calc_rent(val_per_sqm, pyeong):
    sqm = pyeong * 3.306
    monthly = val_per_sqm * 1000 * sqm
    return int(monthly), int(monthly * 12)


def build_reasons(row, age_targets, gender, schedule, desired_monthly):
    reasons, warnings = [], []
    p2030 = float(row.get('비중_2030') or 0)
    pm    = float(row.get('점포당매출') or 0)
    comp  = float(row.get('경쟁강도')  or 0.5)

    if '2030' in age_targets or '20대' in age_targets or '30대' in age_targets:
        if p2030 >= 0.5:
            reasons.append(f"2030 매출 비중 {p2030*100:.0f}%로 높음")
        else:
            warnings.append(f"2030 고객 비중 {p2030*100:.0f}%로 낮음")
    else:
        if p2030 <= 0.4:
            reasons.append(f"중장년 고객 비중 높음 ({(1-p2030)*100:.0f}%)")

    if pm > 0:
        if pm >= desired_monthly:
            reasons.append(f"점포당 평균 매출 {pm/1e4:.0f}만원으로 목표 달성 가능")
        elif pm >= desired_monthly * 0.7:
            reasons.append(f"점포당 평균 매출 {pm/1e4:.0f}만원 (목표의 {pm/desired_monthly*100:.0f}%)")
        else:
            warnings.append(f"점포당 매출 {pm/1e4:.0f}만원으로 목표 대비 낮음")

    if comp <= 0.25:
        reasons.append("경쟁 밀도 낮은 블루오션 상권")
    elif comp >= 0.75:
        warnings.append("경쟁이 치열한 레드오션 상권")

    if schedule == '주중' and float(row.get('주중비중') or 0.5) >= 0.6:
        reasons.append(f"주중 매출 집중 상권 ({row['주중비중']*100:.0f}%)")
    elif schedule == '주말' and float(row.get('주말비중') or 0.5) >= 0.5:
        reasons.append(f"주말 매출 활성 상권 ({row['주말비중']*100:.0f}%)")

    if gender == '여성' and float(row.get('여성비중') or 0.5) >= 0.55:
        reasons.append(f"여성 고객 비중 {row['여성비중']*100:.0f}%")
    elif gender == '남성' and float(row.get('남성비중') or 0.5) >= 0.55:
        reasons.append(f"남성 고객 비중 {row['남성비중']*100:.0f}%")

    return reasons, warnings


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
<div class="hero">
  <h1>🏪 소상공인 창업 입지 추천</h1>
  <p>서울시 상권 빅데이터 기반 · 당신에게 맞는 창업 입지 TOP 5를 추천해드립니다</p>
</div>
""", unsafe_allow_html=True)

    # 데이터 로딩 (구글 드라이브 → 캐시)
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.info("📥 데이터를 불러오는 중입니다... 첫 실행 시 30초 내외 소요됩니다.")
        with st.spinner(""):
            summary, rent_df, industry_list = load_data()
    progress_placeholder.empty()

    if summary is None:
        st.error("데이터 로딩에 실패했습니다. 잠시 후 새로고침 해주세요.")
        st.stop()

    # 요약 지표
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 상권 수",  f"{summary['상권_코드'].nunique():,}개")
    c2.metric("분석 업종 수",  f"{summary['서비스_업종_코드_명'].nunique():,}개")
    c3.metric("분석 레코드",   f"{len(summary):,}건")
    has_gu = '자치구명' in summary.columns
    c4.metric("지역 정보",     "자치구·행정동 포함" if has_gu else "상권명 기준")

    st.markdown("---")
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
                min_value=5, max_value=50, value=10, step=5
            )
            industry = st.selectbox("희망 업종", options=industry_list)

        with col_r:
            st.markdown("### 👥 타겟 고객층")
            age_targets = st.multiselect(
                "타겟 연령대 (복수 선택 가능)",
                options=["2030", "40대", "50대 이상"],
                default=["2030"]
            )
            gender = st.radio("타겟 성별", options=["무관", "남성", "여성"], horizontal=True)
            schedule = st.radio(
                "주요 운영 패턴", options=["무관", "주중", "주말"], horizontal=True
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

    if not submitted:
        return

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
        rank      = rank_i + 1
        score_pct = int(row['총점'] * 100)
        pm        = float(row.get('점포당매출') or 0)
        p2030     = float(row.get('비중_2030')  or 0)
        comp      = float(row.get('경쟁강도')   or 0)
        opbiz     = float(row.get('개업률')     or 0)
        clbiz     = float(row.get('폐업률')     or 0)
        n_stor    = int(row.get('총점포수')      or 0)
        gu_nm     = row.get('자치구명', '')  or ''
        dong_nm   = row.get('행정동명', '')  or ''
        location_sub = f"{gu_nm} {dong_nm}".strip() if gu_nm or dong_nm else row.get('상권_구분_코드_명', '')

        reasons, warnings_list = build_reasons(row, age_targets, gender, schedule, desired_monthly)

        # 임대료
        rent_val = get_rent_for_district(rent_df, row['상권_코드_명'])
        if rent_val:
            monthly_rent, annual_rent = calc_rent(rent_val, pyeong)
            rent_ratio = monthly_rent / (capital * 10000) * 100
            rent_html = (
                f'<div class="rent-box">'
                f'<strong>💰 임대료 추산 ({pyeong}평 기준)</strong>'
                f' &nbsp;—&nbsp; 월 <strong>{monthly_rent:,.0f}원</strong>'
                f' &nbsp;|&nbsp; 연 <strong>{annual_rent:,.0f}원</strong>'
                f' &nbsp;|&nbsp; 자본금 대비 <strong>{rent_ratio:.1f}%</strong>'
                f'</div>'
            )
        else:
            rent_html = (
                '<div class="rent-box" style="color:#999;">'
                '💰 임대료 데이터 매칭 없음</div>'
            )

        # 희망매출 달성 가능성
        if pm > 0:
            ratio = pm / desired_monthly * 100
            ac_color = '#2e7d32' if ratio >= 100 else ('#f57f17' if ratio >= 70 else '#c62828')
            ac_label = (f'✅ 달성 가능 ({ratio:.0f}%)' if ratio >= 100
                        else f'⚠️ 달성 도전 ({ratio:.0f}%)' if ratio >= 70
                        else f'❌ 달성 어려움 ({ratio:.0f}%)')
            achieve_html = (
                f'<div class="achieve-box">'
                f'<strong>📈 희망매출 달성 가능성:</strong> '
                f'<span style="color:{ac_color}; font-weight:600;">{ac_label}</span>'
                f' &nbsp;(평균 점포당 매출 <strong>{pm/1e4:.0f}만원</strong>'
                f' vs 목표 <strong>{desired_sales:,}만원</strong>)'
                f'</div>'
            )
        else:
            achieve_html = '<div class="achieve-box" style="color:#999;">📈 매출 데이터 없음</div>'

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
      <div style="font-size:0.85rem; color:#718096;">{location_sub}</div>
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

    # 비교 테이블
    st.markdown("---")
    st.markdown("### 📊 추천 상권 비교표")

    display_cols = ['상권_코드_명', '상권_구분_코드_명', '총점포수',
                    '점포당매출', '비중_2030', '경쟁강도', '개업률', '폐업률', '총점']
    if '자치구명' in results.columns:
        display_cols.insert(2, '자치구명')
    disp = results[[c for c in display_cols if c in results.columns]].copy()
    disp.rename(columns={
        '상권_코드_명':     '상권명',
        '상권_구분_코드_명': '상권유형',
        '자치구명':         '자치구',
        '점포당매출':       '점포당매출(만원)',
        '비중_2030':        '2030비중(%)',
        '경쟁강도':         '경쟁강도(%)',
        '총점':             '적합도점수',
    }, inplace=True)
    if '점포당매출(만원)' in disp.columns:
        disp['점포당매출(만원)'] = (disp['점포당매출(만원)'] / 1e4).round(0).astype('Int64')
    if '2030비중(%)' in disp.columns:
        disp['2030비중(%)'] = (disp['2030비중(%)'] * 100).round(1)
    if '경쟁강도(%)' in disp.columns:
        disp['경쟁강도(%)'] = (disp['경쟁강도(%)'] * 100).round(1)
    if '적합도점수' in disp.columns:
        disp['적합도점수'] = (disp['적합도점수'] * 100).round(1)
    disp.index = range(1, len(disp) + 1)
    st.dataframe(disp, use_container_width=True)

    # 점수 구성
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

    st.markdown("---")
    st.caption(
        "📊 데이터 출처: 서울시 열린데이터광장 상권분석서비스 (2025) · 한국감정원 소규모 상가 임대동향 | "
        "⚖️ 가중치: 연령 30% · 업종수요 25% · 시간패턴 20% · 성별 10% · 경쟁효율 15%"
    )


if __name__ == "__main__":
    main()
