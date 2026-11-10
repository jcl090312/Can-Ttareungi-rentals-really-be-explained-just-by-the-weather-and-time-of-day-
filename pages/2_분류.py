import streamlit as st
import pandas as pd
from modeling import data,columns,pipe,LogisticRegression,DummyClassifier,accuracy_score,f1_score,confusion_matrix,notes
st.set_page_config(page_title='분류 분석',layout='wide');st.title('2. 분류 · 이용량이 높은 시간일까?')
h,(train,val,test)=data();threshold=train['대여량'].median()
st.write(f'학습 기간 중앙값 {threshold:,.0f}건 초과를 높은 수요(1), 이하를 낮은 수요(0)로 정의합니다. 실제 재고 부족이나 혼잡 판정은 아닙니다.')
cols=columns('시간대+날씨+요일');records=[];models=[]
if (train['대여량']>threshold).nunique()<2:st.warning('학습 자료에 두 범주가 없어 분류할 수 없습니다.');st.stop()
for name,model in [('다수 범주 기준',DummyClassifier(strategy='most_frequent')),('로지스틱 회귀',LogisticRegression(max_iter=2000,C=1))]:
 m=pipe(model);m.fit(train[cols],(train['대여량']>threshold).astype(int));y=(val['대여량']>threshold).astype(int);p=m.predict(val[cols]);records.append({'모델':name,'정확도':accuracy_score(y,p),'Macro F1':f1_score(y,p,labels=[0,1],average='macro',zero_division=0)});models.append((name,m))
scores=pd.DataFrame(records).sort_values('Macro F1',ascending=False);st.subheader('검증 성능');st.dataframe(scores,hide_index=True)
name=scores.iloc[0]['모델'];m=dict(models)[name];fit=pd.concat([train,val]);m.fit(fit[cols],(fit['대여량']>threshold).astype(int));y=(test['대여량']>threshold).astype(int);p=m.predict(test[cols])
st.write(f'검증 F1로 선택: {name}');st.write({'최종 정확도':accuracy_score(y,p),'최종 Macro F1':f1_score(y,p,labels=[0,1],average='macro',zero_division=0)})
st.write('혼동행렬: 행은 실제, 열은 예측');st.dataframe(pd.DataFrame(confusion_matrix(y,p,labels=[0,1]),index=['실제 낮음','실제 높음'],columns=['예측 낮음','예측 높음']))
notes()
