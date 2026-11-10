from pathlib import Path
import io
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

def merge_weather(hourly, weather):
    required = {'datetime', 'temperature', 'rainfall'}
    if not required.issubset(weather.columns):
        raise ValueError('필수 열: datetime, temperature, rainfall')
    weather = weather.copy()
    dates = pd.to_datetime(weather['datetime'], errors='coerce')
    if dates.isna().any():
        raise ValueError('날씨 파일에 읽을 수 없는 날짜가 있습니다.')
    if dates.dt.tz is not None:
        raise ValueError('datetime은 시간대 표기 없는 한국 현지 시각으로 입력하세요.')
    weather['시각'] = dates.dt.floor('h')
    if weather['시각'].duplicated().any():
        raise ValueError('날씨는 한 관측 기준으로 시간당 한 행만 입력하세요.')
    for col in ['temperature', 'rainfall']:
        weather[col] = pd.to_numeric(weather[col], errors='coerce').replace([float('inf'), -float('inf')], float('nan'))
    return hourly.reset_index().merge(weather[['시각', 'temperature', 'rainfall']], on='시각', how='inner', validate='one_to_one')


def read_weather(source):
    raw = source.read_bytes() if isinstance(source, Path) else source.getvalue()
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('cp949')
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith(('날짜,', 'date,', 'datetime,'))), None)
    if start is None:
        raise ValueError('날짜, date 또는 datetime으로 시작하는 열 이름 행이 필요합니다.')
    w = pd.read_csv(io.StringIO('\n'.join(lines[start:])), dtype=str)
    w.columns = w.columns.str.strip()
    w = w.rename(columns={'날짜': 'date', '평균기온(℃)': 'temperature', '최저기온(℃)': 'temperature_min', '최고기온(℃)': 'temperature_max', '일강수량(mm)': 'rainfall', '강수량(mm)': 'rainfall'})
    key = 'date' if 'date' in w.columns else 'datetime'
    w[key] = w[key].fillna('').str.strip()
    w = w.loc[w[key] != ''].copy()
    if 'temperature' not in w:
        raise ValueError('평균기온(℃) 또는 temperature 열이 필요합니다.')
    for col in ['temperature', 'temperature_min', 'temperature_max', 'rainfall']:
        if col in w:
            w[col] = pd.to_numeric(w[col], errors='coerce').replace([float('inf'), -float('inf')], float('nan'))
    if 'rainfall' not in w:
        w['rainfall'] = float('nan')
    if key == 'date':
        w['date'] = pd.to_datetime(w['date'], errors='coerce').dt.normalize()
        if w['date'].isna().any() or w['date'].duplicated().any():
            raise ValueError('일별 날씨 날짜가 잘못되었거나 중복입니다. 하루 한 행만 사용하세요.')
    return w, key == 'date'


def merge_daily(hourly, weather):
    daily = hourly.groupby(hourly.index.normalize()).agg(건수=('건수', 'sum'), 관측시간수=('건수', 'size'))
    daily.index.name = 'date'
    return daily.reset_index().merge(weather, on='date', how='inner', validate='one_to_one')


def rain_groups(merged):
    rain = merged.loc[merged['rainfall'].notna() & (merged['rainfall'] >= 0)].copy()
    rain['강수 여부'] = rain['rainfall'].gt(0).map({True: '강수량 > 0mm', False: '기록된 강수량 0mm'})
    return rain


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
    st.subheader('① 시간대별 평균 대여량 · 막대그래프')
    by_hour = h.groupby('시간대')['건수'].mean().reindex(range(24)).rename('평균 대여량').rename_axis('시간대').reset_index()
    fig = px.bar(by_hour, x='시간대', y='평균 대여량', labels={'평균 대여량': '관측 시간당 평균 대여량(건)'})
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption('각 시간대의 총 대여량을 그 시간대가 관측된 날짜 수로 나눕니다. 기록이 없는 시간은 0건으로 간주하지 않습니다.')
    st.subheader('④ 요일 × 시간대 · 히트맵')
    heat = h.pivot_table(index='요일번호', columns='시간대', values='건수', aggfunc='mean').reindex(index=range(7), columns=range(24))
    heat.index = ['월', '화', '수', '목', '금', '토', '일']
    fig = px.imshow(heat, aspect='auto', color_continuous_scale='YlGnBu',
                    labels={'x': '시간대', 'y': '요일', 'color': '평균 대여량(건)'})
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption('같은 요일·시간의 관측값을 평균합니다. 빈 칸은 관측 없음입니다. 같은 시간에도 평일과 주말의 이용 패턴이 다른지 살펴보세요.')
    st.caption('두 그래프 모두 부분 업로드와 불완전한 시간 기록에 영향을 받습니다.')
with tabs[1]:
    st.info('대여소 순위는 업로드된 전체 기간 기준입니다. 왼쪽 날짜 필터는 적용되지 않습니다.')
    n = st.slider('표시할 상위 대여소 수', 5, 50, 15)
    top = stations.head(n).reset_index()
    top['대여소'] = top['대여 대여소명'] + ' (' + top['대여대여소ID'] + ')'
    st.plotly_chart(px.bar(top.sort_values('대여량'), x='대여량', y='대여소', orientation='h'), use_container_width=True)
    st.caption('대여량은 관측된 이용 건수입니다. 자전거 부족으로 발생한 미충족 수요나 재고량을 의미하지 않습니다.')
with tabs[2]:
    st.write('일별 기온 파일은 하루 총대여량과, 시간별 날씨 파일은 시간당 대여량과 비교합니다.')
    uploaded = st.file_uploader('날씨 CSV (기상청 기온분석 원본도 지원)', type=['csv'])
    weather_path = ROOT / 'data' / 'weather.csv'
    source = uploaded if uploaded is not None else (weather_path if weather_path.is_file() else None)
    if source is None:
        st.info('data/weather.csv를 넣거나 위에서 날씨 파일을 올려주세요.')
    else:
        try:
            weather, is_daily = read_weather(source)
            if is_daily:
                merged = merge_daily(h, weather)
                st.write(f'날짜가 일치하는 기록: {len(merged)}일 / 날씨 {len(weather)}일')
                incomplete = int((merged['관측시간수'] < 24).sum())
                st.caption(f'24개 시간대가 관측되지 않은 날짜: {incomplete}일. 24시간이 있어도 분할 파일 전체의 완전성을 보장하지는 않습니다.')
                if st.checkbox('24개 시간대가 관측된 날짜만 비교', value=True):
                    merged = merged.loc[merged['관측시간수'] == 24].copy()
                x_label, y_label, time_key, unit = '일평균기온(℃)', '하루 총대여량(건)', 'date', '일'
                merged['요일'] = merged['date'].dt.dayofweek.map(dict(enumerate(['월','화','수','목','금','토','일'])))
                color = '요일'
            else:
                merged = merge_weather(h, weather)
                x_label, y_label, time_key, unit = '기온(℃)', '시간당 대여량(건)', '시각', '시간'
                color = '시간대'
                st.write(f'시간이 일치하는 기록: {len(merged)} / {len(h)}시간')
            st.subheader('② 기온–대여량 · 산점도')
            valid = merged.dropna(subset=['temperature', '건수'])
            if valid.empty:
                st.info('필터 조건을 만족하는 기온·대여량 쌍이 없습니다. 기간과 전체 데이터 업로드 여부를 확인하세요.')
            else:
                st.plotly_chart(px.scatter(valid, x='temperature', y='건수', color=color, hover_data=[time_key],
                                           labels={'temperature': x_label, '건수': y_label}), use_container_width=True)
                st.caption(f'점 하나는 1{unit}입니다. 유효 관측: {len(valid)}{unit}.')
                if len(valid) >= 3 and valid['temperature'].nunique() > 1 and valid['건수'].nunique() > 1:
                    st.write(f'피어슨 상관계수: {valid["temperature"].corr(valid["건수"]):.3f}')
            st.subheader('③ 기록된 강수량 구분별 대여량 · 상자그림')
            st.caption(f'선택·결합된 자료 중 강수량 빈칸: {merged["rainfall"].isna().sum()}개. 빈칸은 0으로 바꾸지 않고 상자그림에서 제외합니다.')
            rain = rain_groups(merged)
            if rain.empty:
                st.info('현재 자료에는 비교 가능한 강수량이 없습니다. 기온 파일만으로 비가 왔는지 알 수 없습니다. 일별 자료에는 같은 날짜의 일강수량(mm) 또는 rainfall 열을 추가하세요.')
            else:
                st.plotly_chart(px.box(rain, x='강수 여부', y='건수', color='강수 여부', points='outliers',
                                       category_orders={'강수 여부': ['기록된 강수량 0mm','강수량 > 0mm']}, labels={'건수': y_label}), use_container_width=True)
                st.dataframe(rain.groupby('강수 여부')['건수'].agg(['count','median','mean']).rename(columns={'count': f'관측 {unit} 수','median':'중앙값','mean':'평균'}))
                st.caption('기록된 0mm와 0mm 초과를 비교합니다. 0mm만으로 무강수를 단정하지 않습니다. 빈칸·음수는 제외합니다. 상자는 가운데 50%, 내부 선은 중앙값입니다.')
                st.caption('두 집단의 표본 수가 작거나 다를 수 있습니다. 평균뿐 아니라 중앙값·관측 수를 함께 보고, 관측값만으로 해석하세요.')
                if rain['강수 여부'].nunique() < 2:
                    st.info('한 범주만 존재해 두 집단 비교는 불가능합니다.')
            st.warning('기온과 대여량의 상관관계는 인과관계를 뜻하지 않습니다. 요일 등의 영향을 함께 살펴보세요. 일평균기온을 시간별 관측기온으로 해석하지 않습니다.')
            st.download_button('날씨 결합 자료 CSV', merged.to_csv(index=False).encode('utf-8-sig'), 'weather_joined.csv', 'text/csv')
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
