from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from rain_loader import attach_rain, rain_files
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix, silhouette_score
from sklearn.cluster import KMeans
ROOT = Path(__file__).resolve().parent

@st.cache_data(show_spinner='모델용 시간별 대여량 집계 중…', max_entries=2)
def load_data(signature):
    counts=[]
    for name, _, _ in signature:
        if not name.endswith('.gz'): continue
        for c in pd.read_csv(name, encoding='utf-8-sig',usecols=['대여일시'],chunksize=100000):
            dt=pd.to_datetime(c['대여일시'],errors='coerce')
            counts.append(dt.dropna().dt.floor('h').value_counts())
    h=pd.concat(counts).groupby(level=0).sum().sort_index().rename('대여량').rename_axis('시각').reset_index()
    h['date']=h['시각'].dt.normalize()
    observed=h.groupby('date').size()
    h=h[h['date'].isin(observed[observed==24].index)].copy()
    w=pd.read_csv(ROOT/'data/weather.csv',encoding='utf-8-sig')
    w['date']=pd.to_datetime(w['date'],errors='raise').dt.normalize()
    if w['date'].duplicated().any(): raise ValueError('날씨 날짜가 중복입니다.')
    for col in ['temperature','rainfall']:
        if col not in w: w[col]=np.nan
        w[col]=pd.to_numeric(w[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
    w.loc[w.rainfall<0,'rainfall']=np.nan
    w,_=attach_rain(w,ROOT)
    h=h.merge(w[['date','temperature','rainfall']],on='date',how='left',validate='many_to_one')
    for col,n,values in [('hour',24,h['시각'].dt.hour),('weekday',7,h['시각'].dt.dayofweek)]:
        for i in range(n): h[f'{col}_{i}']=(values==i).astype(float)
    return h.sort_values('시각').reset_index(drop=True)

def data():
    files=sorted((ROOT/'data').glob('bike_part_*.csv.gz'))
    if not files or not (ROOT/'data/weather.csv').exists():
        st.error('data 폴더에 전체 bike_part_*.csv.gz와 결합된 weather.csv를 넣으세요.');st.stop()
    try:
        paths=files+[ROOT/'data/weather.csv']+rain_files(ROOT)
        h=load_data(tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths))
    except Exception as e: st.error(str(e));st.stop()
    days=sorted(h.date.unique())
    if len(days)<15:
        st.warning(f'24개 시간대가 관측된 날짜가 {len(days)}일입니다. 모델 비교에는 최소 15일의 전체 분할 파일이 필요합니다.');st.stop()
    a=int(len(days)*.6); b=int(len(days)*.8)
    parts=[h[h.date.isin(d)] for d in [days[:a],days[a:b],days[b:]]]
    st.caption('날짜 순서로 앞 60% 학습 / 다음 20% 검증·선택 / 마지막 20% 최종 평가. 같은 날짜는 분할하지 않습니다.')
    st.write(pd.DataFrame([{'용도':label,'시작':p.date.min().date(),'종료':p.date.max().date(),'시간 수':len(p)} for label,p in zip(['학습','검증','최종 평가'],parts)]))
    st.info('일별 날씨가 같은 날 24개 시간 행에 반복됩니다. 독립적인 날씨 관측은 날짜 수만큼이며, 한 달 결과를 다른 계절에 일반화할 수 없습니다. 24시간 관측 조건도 파일 완전성을 보장하지 않습니다.')
    return h,parts

def columns(kind):
    cols=[f'hour_{i}' for i in range(24)]
    if kind!='시간대': cols+=['temperature','rainfall']
    if kind=='시간대+날씨+요일':cols += [f'weekday_{i}' for i in range(7)]
    return cols

def pipe(model):
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),model)

def metrics(y,p):
    return {'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),'R²':r2_score(y,p)}

@st.cache_data(show_spinner='두 회귀 모델을 학습·평가하고 있습니다…', max_entries=2)
def regression(parts):
    train,val,test=parts
    # 기준 범주를 제외하여 절편과 범주 변수 사이의 중복을 피합니다.
    cols=[c for c in columns('시간대+날씨+요일') if c not in ['hour_0','weekday_0']]
    candidates=[('선형 회귀',LinearRegression(fit_intercept=True)),
                ('랜덤포레스트 회귀',RandomForestRegressor(n_estimators=150,max_depth=8,min_samples_leaf=12,random_state=42,n_jobs=1))]
    records=[]; fitted={}
    for name,model in candidates:
        m=pipe(model); m.fit(train[cols],train['대여량'])
        pred=np.maximum(0,m.predict(val[cols]))
        records.append({'모델':name,'구분':'선택 후보',**metrics(val['대여량'],pred)})
        fitted[name]=m
    # 동점이면 표 순서에 따라 단순한 선형 회귀를 선택합니다.
    best=min(records,key=lambda row:row['MAE'])['모델']
    records.append({'모델':'평균 기준','구분':'비교용 베이스라인',**metrics(val['대여량'],np.repeat(train['대여량'].mean(),len(val)))})
    scores=pd.DataFrame(records).sort_values('MAE',kind='stable')
    fit=pd.concat([train,val]);m=fitted[best];m.fit(fit[cols],fit['대여량'])
    p=np.maximum(0,m.predict(test[cols]));baseline=np.repeat(fit['대여량'].mean(),len(test))
    out=test[['시각','대여량']].copy();out['예측']=p;out['잔차']=out['대여량']-p
    return scores,best,metrics(test['대여량'],p),metrics(test['대여량'],baseline),out


def explain_model(name):
    st.subheader('선택된 모델의 원리와 설정')
    if name=='선형 회귀':
        st.write('시간대·기온·강수량·요일 등의 값에 학습한 계수를 곱해 더하여 대여량을 예측합니다. 실제 대여량과 예측값의 제곱 오차 합을 최소화하도록 계수를 찾습니다.')
        st.latex(r'\hat{y}=b_0+b_1x_1+\cdots+b_px_p')
        st.write('fit_intercept=True: 기본 수준을 나타내는 절편도 학습합니다. 계수와 절편은 데이터로 학습하는 파라미터입니다. 이 모델에는 Ridge의 alpha나 나무 깊이 설정이 없습니다.')
        st.caption('0시는 시간대 기준, 월요일은 요일 기준으로 두었습니다. 표준화된 입력을 사용하므로 계수를 그대로 “기온 1℃당 변화”로 읽으면 안 됩니다. 상호작용 항이 없어 요일에 따라 출퇴근 봉우리 모양이 달라지는 현상을 충분히 반영하지 못할 수 있습니다.')
    else:
        st.write('여러 결정나무가 조건으로 데이터를 나누어 각각 대여량을 예측하고, 150개 나무의 예측을 평균합니다. 개별 나무의 최종 구간은 그 구간 학습 표본의 평균값으로 예측합니다.')
        st.dataframe(pd.DataFrame([
            {'설정':'n_estimators','값':'150','뜻':'함께 예측하는 나무 수'},
            {'설정':'max_depth','값':'8','뜻':'각 나무의 최대 깊이'},
            {'설정':'min_samples_leaf','값':'12','뜻':'최종 구간에 필요한 최소 시간별 학습 행 수'},
            {'설정':'random_state','값':'42','뜻':'무작위 과정을 재현하기 위한 고정값'},
            {'설정':'n_jobs','값':'1','뜻':'병렬 실행 작업 수'}]),hide_index=True)
        st.caption('앞의 세 개는 모델 복잡도를 조절하며, 마지막 두 개는 재현성과 실행 설정입니다. 현재 값은 고정한 후보 설정이며 최적값을 탐색한 결과는 아닙니다.')
    st.caption('두 후보에 같은 입력 변수와 전처리를 사용합니다. 예측 후 음수는 0건으로 제한하며 평가에도 같은 처리를 적용합니다. 후보 선택은 검증 MAE, 마지막 기간은 선택 후 평가에만 사용합니다.')


def notes():
    st.caption('강수 빈칸은 원본에서 보존합니다. 모델 입력에서만 학습 자료의 중앙값과 결측 표시를 사용하며, 전체가 결측인 열의 내부 대체값은 관측 강수량을 뜻하지 않습니다. 0을 결측으로 보지 않습니다. 대여량 이상 후보도 자동 삭제하지 않습니다.')
    st.warning('관측된 일평균기온·일강수량을 사용한 사후 예측 비교입니다. 실제 미래 예측에는 당시 이용 가능한 날씨 예보가 필요합니다. 예측 성능은 인과관계의 증거가 아닙니다.')
