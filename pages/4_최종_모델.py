import streamlit as st
from modeling import data,regression,notes
st.set_page_config(page_title='최종 모델',layout='wide');st.title('4. 최종 선택 · 회귀')
st.write('연구 목표가 대여량이라는 수치를 설명·예측하는 것이므로 회귀를 주 모델로 선택합니다. 분류는 높은 이용량 시간 구분, 군집은 하루 패턴 탐색을 위한 보조 분석입니다. 서로 다른 지표의 숫자를 비교해 승자를 정하지 않습니다.')
h,parts=data();scores,name,result,baseline,out=regression(parts)
st.success(f'현재 데이터의 검증 기간에서 선택된 회귀 후보: {name}')
st.dataframe(scores,hide_index=True);st.write('선택 후 최종 평가:',result)
gain=baseline['MAE']-result['MAE']
if gain>0:st.write(f'최종 평가에서 평균 기준보다 MAE가 {gain:,.1f}건 작습니다.')
else:st.warning('최종 평가에서 평균 기준을 개선하지 못했습니다. 검증 기간의 선택이 미래 기간에서도 우수하다고 볼 수 없습니다.')
st.line_chart(out.set_index('시각')[['대여량','예측']])
st.write('보고서 판단 순서: 검증 표에서 날씨·요일 추가가 MAE를 낮추는지 확인 → 최종 기간에서 기준 대비 개선 확인 → 한 달·강수 결측·관측 날씨 사용의 한계를 함께 기재.')
notes()
