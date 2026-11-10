from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
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
    h=h.merge(w[['date','temperature','rainfall']],on='date',how='left',validate='many_to_one')
    for col,n,values in [('hour',24,h['시각'].dt.hour),('weekday',7,h['시각'].dt.dayofweek)]:
        for i in range(n): h[f'{col}_{i}']=(values==i).astype(float)
    return h.sort_values('시각').reset_index(drop=True)

def data():
    files=sorted((ROOT/'data').glob('bike_part_*.csv.gz'))
    if not files or not (ROOT/'data/weather.csv').exists():
        st.error('data 폴더에 전체 bike_part_*.csv.gz와 결합된 weather.csv를 넣으세요.');st.stop()
    try:
        paths=files+[ROOT/'data/weather.csv']
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

def regression(parts):
    train,val,test=parts; records=[]; candidates=[]
    for name,kind,model in [('평균 기준','시간대',DummyRegressor()),('Ridge 시간대','시간대',Ridge(alpha=10)),('Ridge 시간대+날씨','시간대+날씨',Ridge(alpha=10)),('Ridge 시간대+날씨+요일','시간대+날씨+요일',Ridge(alpha=10)),('랜덤포레스트 전체','시간대+날씨+요일',RandomForestRegressor(n_estimators=150,max_depth=8,min_samples_leaf=12,random_state=42,n_jobs=1))]:
        cols=columns(kind); m=pipe(model);m.fit(train[cols],train['대여량'])
        pred=np.maximum(0,m.predict(val[cols]));records.append({'모델':name,**metrics(val['대여량'],pred)})
        candidates.append((name,cols,m))
    scores=pd.DataFrame(records).sort_values('MAE');best=scores.iloc[0]['모델']
    name,cols,m=next(c for c in candidates if c[0]==best)
    fit=pd.concat([train,val]);m.fit(fit[cols],fit['대여량']);p=np.maximum(0,m.predict(test[cols]))
    baseline=np.repeat(fit['대여량'].mean(),len(test))
    out=test[['시각','대여량']].copy();out['예측']=p;out['잔차']=out['대여량']-p
    return scores,name,metrics(test['대여량'],p),metrics(test['대여량'],baseline),out

def notes():
    st.caption('강수 빈칸은 원본에서 보존합니다. 모델 입력에서만 학습 자료의 중앙값과 결측 표시를 사용하며, 전체가 결측인 열의 내부 대체값은 관측 강수량을 뜻하지 않습니다. 0을 결측으로 보지 않습니다. 대여량 이상 후보도 자동 삭제하지 않습니다.')
    st.warning('관측된 일평균기온·일강수량을 사용한 사후 예측 비교입니다. 실제 미래 예측에는 당시 이용 가능한 날씨 예보가 필요합니다. 예측 성능은 인과관계의 증거가 아닙니다.')
