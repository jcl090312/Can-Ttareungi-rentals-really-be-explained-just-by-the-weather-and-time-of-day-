from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='따릉연구소', page_icon='🚲', layout='wide')
ROOT = Path(__file__).resolve().parent
COLS = ['대여일시', '대여대여소ID', '대여 대여소명', '이용시간(분)', '이용거리(M)']

@st.cache_data(show_spinner='분할 파일을 순서대로 집계하고 있습니다…', max_entries=2)
def aggregate(signature):
    hours, stations = [], []
    total = invalid = zero = 0
    for filename, size, modified in signature:
        for chunk in pd.read_csv(filename, compression='infer', encoding='utf-8-sig',
                                 usecols=COLS, dtype=str, chunksize=100000):
            total += len(chunk)
            dt = pd.to_datetime(chunk['대여일시'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            invalid += int(dt.isna().sum())
            duration = pd.to_numeric(chunk['이용시간(분)'], errors='coerce')
            distance = pd.to_numeric(chunk['이용거리(M)'], errors='coerce')
            zero += int(((duration == 0) | (distance == 0)).sum())
            x = pd.DataFrame({'시각': dt.dt.floor('h'), '건수': 1,
                              '시간합': duration.where(duration >= 0),
                              '거리합': distance.where(distance >= 0)})
            x['시간수'] = x['시간합'].notna().astype(int)
            x['거리수'] = x['거리합'].notna().astype(int)
            hours.append(x.dropna(subset=['시각']).groupby('시각').sum())
            s = chunk.loc[dt.notna(), ['대여대여소ID', '대여 대여소명']].fillna('미상')
            stations.append(s.groupby(['대여대여소ID', '대여 대여소명']).size().rename('대여량'))
    if not hours:
        raise ValueError('읽을 데이터 행이 없습니다.')
    hourly = pd.concat(hours).groupby(level=0).sum().sort_index()
    station = pd.concat(stations).groupby(level=[0, 1]).sum().sort_values(ascending=False)
    return hourly, station, total, invalid, zero

st.title('🚲 따릉연구소')
st.caption('따릉이 대여량은 정말 날씨와 시간대만으로 설명되는가?')
files = sorted((ROOT / 'data').glob('bike_part_*.csv.gz'))
if not files:
    st.error('data 폴더에 bike_part_0001.csv.gz 등의 분할 파일을 넣어주세요.')
    st.code('main.py\nrequirements.txt\ndata/\n  bike_part_0001.csv.gz\n  bike_part_0002.csv.gz')
    st.stop()
st.sidebar.header('데이터 범위')
st.sidebar.write(f'발견한 분할 파일: {len(files)}개')
st.sidebar.caption('각 분할 파일을 한 번씩만 넣으세요. 같은 파일의 복사본도 합산됩니다.')
if st.sidebar.button('캐시 지우고 다시 읽기'):
    aggregate.clear()
try:
    signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files)
    hourly, stations, total, invalid, zero = aggregate(signature)
except Exception as e:
    st.error(f'데이터를 읽지 못했습니다: {e}')
    st.stop()
if hourly.empty:
    st.error('해석 가능한 대여일시가 없습니다.')
    st.stop()
st.warning('분할 파일 전체 업로드 여부는 자동 확인할 수 없습니다. 일부 파일만 있으면 결과도 일부 기록의 집계입니다. 누락된 시간은 자동으로 0건 처리하지 않습니다.')
a, b, c, d = st.columns(4)
a.metric('읽은 기록', f'{total:,}건')
b.metric('날짜 해석 실패', f'{invalid:,}건')
c.metric('관측 대여소 ID', f'{stations.reset_index()["대여대여소ID"].nunique():,}개')
d.metric('시간 또는 거리 0인 기록', f'{zero:,}건')
st.caption(f'관측 범위: {hourly.index.min()} ~ {hourly.index.max()} · 0인 기록은 기본 집계에 포함합니다.')
selected = st.sidebar.date_input('시간 분석 기간', value=(hourly.index.min().date(), hourly.index.max().date()),
                                min_value=hourly.index.min().date(), max_value=hourly.index.max().date())
if len(selected) != 2:
    st.info('시작일과 종료일을 모두 선택하세요.')
    st.stop()
h = hourly.loc[(hourly.index.date >= selected[0]) & (hourly.index.date <= selected[1])].copy()
if h.empty:
    st.info('선택한 기간에 관측 기록이 없습니다.')
    st.stop()
h['시간대'] = h.index.hour
h['요일번호'] = h.index.dayofweek
tabs = st.tabs(['시간·요일', '대여소', '날씨 비교', '해석·다운로드'])
with tabs[0]:
    daily = h.groupby(h.index.date)['건수'].sum().rename_axis('날짜').reset_index()
    st.plotly_chart(px.line(daily, x='날짜', y='건수', markers=True, title='일별 대여 기록 수'), use_container_width=True)
    metric = st.radio('시간대 비교 방식', ['관측 시간당 평균', '합계'], horizontal=True)
    grouped = h.groupby('시간대')['건수']
    by_hour = (grouped.mean() if metric == '관측 시간당 평균' else grouped.sum()).rename('대여량').reset_index()
    st.plotly_chart(px.bar(by_hour, x='시간대', y='대여량', title=f'시간대별 {metric}'), use_container_width=True)
    daily['요일'] = pd.to_datetime(daily['날짜']).dt.dayofweek
    weekday = daily.groupby('요일')['건수'].mean().reindex(range(7))
    weekday.index = ['월', '화', '수', '목', '금', '토', '일']
    st.bar_chart(weekday.rename('관측일당 평균 대여량'))
    st.caption('평균은 기록이 존재하는 시간·날짜를 분모로 계산합니다. 부분 업로드, 월초·월말의 불완전한 날짜, 누락 기록에 영향을 받습니다.')
with tabs[1]:
    st.info('대여소 순위는 업로드된 전체 기간 기준입니다. 왼쪽 날짜 필터는 적용되지 않습니다.')
    n = st.slider('표시할 상위 대여소 수', 5, 50, 15)
    top = stations.head(n).reset_index()
    top['대여소'] = top['대여 대여소명'] + ' (' + top['대여대여소ID'] + ')'
    st.plotly_chart(px.bar(top.sort_values('대여량'), x='대여량', y='대여소', orientation='h'), use_container_width=True)
    st.caption('대여량은 관측된 이용 건수입니다. 자전거 부족으로 발생한 미충족 수요나 재고량을 의미하지 않습니다.')
with tabs[2]:
    st.write('첨부된 대여 기록에는 날씨가 없습니다. 서울의 시간별 날씨 CSV를 추가하면 비교할 수 있습니다.')
    st.code('datetime,temperature,rainfall\n2026-06-01 00:00:00,20.1,0')
    st.caption('위 값은 형식 예시이며 실제 관측값이 아닙니다. datetime은 한국 시각, temperature는 ℃, rainfall은 해당 시간의 강수량(mm)으로 준비하세요.')
    uploaded = st.file_uploader('시간별 날씨 CSV', type=['csv'])
    if uploaded:
        try:
            weather = pd.read_csv(uploaded, encoding='utf-8-sig')
            required = {'datetime', 'temperature', 'rainfall'}
            if not required.issubset(weather.columns):
                raise ValueError('필수 열: datetime, temperature, rainfall')
            weather['시각'] = pd.to_datetime(weather['datetime'], errors='coerce').dt.floor('h')
            if weather['시각'].isna().any():
                raise ValueError('날씨 파일에 읽을 수 없는 날짜가 있습니다.')
            if weather['시각'].duplicated().any():
                raise ValueError('같은 시간의 날씨가 여러 행입니다. 서울의 한 관측 기준으로 시간당 한 행만 남기세요.')
            for col in ['temperature', 'rainfall']:
                weather[col] = pd.to_numeric(weather[col], errors='coerce')
            merged = h.reset_index().merge(weather[['시각', 'temperature', 'rainfall']], on='시각', how='inner')
            st.write(f'시간이 일치하는 기록: {len(merged):,} / {len(h):,}시간')
            feature = st.selectbox('날씨 변수', ['temperature', 'rainfall'])
            valid = merged.dropna(subset=[feature, '건수'])
            st.plotly_chart(px.scatter(valid, x=feature, y='건수', color='시간대', title='날씨와 시간별 대여량'), use_container_width=True)
            if len(valid) >= 3 and valid[feature].nunique() > 1 and valid['건수'].nunique() > 1:
                st.metric('피어슨 상관계수', f'{valid[feature].corr(valid["건수"]):.3f}')
            else:
                st.info('상관계수를 계산할 유효한 관측값이나 변동이 부족합니다.')
            st.warning('이 그래프와 상관계수만으로 날씨의 인과효과나 “날씨·시간대만으로 충분하다”는 결론을 내릴 수 없습니다. 요일·계절·지역 등의 영향과 별도 기간 예측 성능을 함께 검토해야 합니다.')
        except Exception as e:
            st.error(f'날씨 파일 확인 필요: {e}')
with tabs[3]:
    st.subheader('수학교육 관점')
    st.write('합계와 평균을 비교하고, 관측일 수가 다른 요일을 공정하게 비교하는 방법을 설명해 보세요. 상관관계와 인과관계도 구분해야 합니다.')
    st.subheader('공학 관점')
    st.write('혼잡 시간과 이용량이 많은 대여소를 찾아 재배치 후보를 제안할 수 있습니다. 최적 배치 결론에는 반납량, 재고, 대여소 용량, 이동 비용이 추가로 필요합니다.')
    st.write('연구의 다음 단계: 전체 파일 완전성 확인 → 날씨 결합 → 시간대만 사용한 모델과 날씨·요일 등을 추가한 모델의 미래 기간 예측 오차 비교.')
    st.download_button('선택 기간 시간별 집계 CSV', h.to_csv(encoding=None).encode('utf-8-sig'), 'hourly_summary.csv', 'text/csv')
    st.download_button('전체 기간 대여소 집계 CSV', stations.to_csv().encode('utf-8-sig'), 'station_summary.csv', 'text/csv')
