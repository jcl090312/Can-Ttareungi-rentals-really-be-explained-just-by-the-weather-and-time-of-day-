import streamlit as st
import pandas as pd
from modeling import data,StandardScaler,KMeans,silhouette_score
st.set_page_config(page_title='군집 분석',layout='wide');st.title('3. 군집 · 비슷한 하루 이용 패턴 찾기')
h,_=data();h['hour']=h['시각'].dt.hour
x=h.pivot(index='date',columns='hour',values='대여량').dropna()
x=x.div(x.sum(axis=1).replace(0,float('nan')),axis=0).dropna()
if len(x)<5:st.warning('군집을 비교할 날짜가 부족합니다.');st.stop()
st.write('하루 24시간 대여량을 하루 총량으로 나눈 비율로 비교합니다. 총량보다 시간대별 패턴을 탐색하며, 전체 날짜를 사용하는 탐색 분석입니다.')
z=StandardScaler().fit_transform(x);results=[];models={}
for k in range(2,min(5,len(x))):
 m=KMeans(n_clusters=k,n_init=20,random_state=42).fit(z)
 if 1<len(set(m.labels_))<len(x):results.append({'군집 수':k,'실루엣 점수':silhouette_score(z,m.labels_)});models[k]=m
if not results:st.info('서로 다른 군집을 구성할 수 없습니다.');st.stop()
scores=pd.DataFrame(results).sort_values('실루엣 점수',ascending=False);st.dataframe(scores,hide_index=True);k=int(scores.iloc[0]['군집 수']);labels=models[k].labels_+1
st.write(f'실루엣 점수로 선택한 군집 수: {k}. 점수가 높을수록 군집 분리가 뚜렷하지만 실제 유형의 정답을 뜻하지 않습니다.')
profiles=x.assign(군집=labels).groupby('군집').mean().T
st.line_chart(profiles);summary=pd.DataFrame({'날짜':x.index,'군집':labels});summary['요일']=summary['날짜'].dt.day_name();st.dataframe(summary,hide_index=True)
st.caption('평일형·주말형이라는 이름은 날짜와 패턴을 확인한 뒤 붙여야 합니다. 군집 번호 자체에는 순서나 우열이 없습니다. 미래 대여량 예측 정확도와 비교할 수 없습니다.')
