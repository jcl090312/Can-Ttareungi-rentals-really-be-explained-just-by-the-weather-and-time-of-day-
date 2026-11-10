import streamlit as st
from modeling import data,regression,notes
st.set_page_config(page_title='회귀 분석',layout='wide')
st.title('1. 회귀 · 대여량은 몇 건일까?')
st.write('목표값은 시간당 대여량입니다. 시간대만 → 날씨 추가 → 요일 추가 순서로 같은 검증 기간의 성능을 비교합니다. MAE·RMSE는 작을수록 좋고 R²는 클수록 좋습니다.')
h,parts=data();scores,name,result,baseline,out=regression(parts)
st.subheader('검증 기간 성능 · 이 표의 MAE로 선택');st.dataframe(scores,hide_index=True)
st.success(f'검증 MAE가 가장 작은 후보: {name}')
st.subheader('선택 완료 후 마지막 기간 평가');st.write(result)
st.write('같은 최종 기간의 평균 기준 모델:',baseline)
st.line_chart(out.set_index('시각')[['대여량','예측']]);st.bar_chart(out.set_index('시각')['잔차'])
st.download_button('최종 기간 예측 CSV',out.to_csv(index=False).encode('utf-8-sig'),'predictions.csv')
notes()
