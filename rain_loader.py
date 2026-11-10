"""기상청 일별 rn CSV를 날짜·지점으로 결합합니다."""
from pathlib import Path
import io
import pandas as pd


def rain_files(root):
    return sorted(set(list((root/'data').glob('rn*.csv')) + list(root.glob('rn*.csv'))))


def attach_rain(weather, root, uploaded=None):
    sources = [uploaded] if uploaded is not None else rain_files(root)
    if not sources:
        return weather, '기온/결합 파일 자체의 rainfall 열 사용 (별도 rn 파일 없음)'
    if len(sources) > 1:
        raise ValueError('rn 파일이 여러 개 있습니다. 강수량 업로드 칸에서 사용할 파일 하나를 선택하세요.')
    source=sources[0]
    raw=source.read_bytes() if isinstance(source,Path) else source.getvalue()
    try: text=raw.decode('utf-8-sig')
    except UnicodeDecodeError: text=raw.decode('cp949')
    lines=text.splitlines()
    start=next((i for i,line in enumerate(lines) if line.strip().startswith(('날짜,','date,'))),None)
    if start is None: raise ValueError('강수 파일에서 날짜 열을 찾지 못했습니다.')
    r=pd.read_csv(io.StringIO('\n'.join(lines[start:])),dtype=str)
    r.columns=r.columns.str.strip()
    r=r.rename(columns={'날짜':'date','강수량(mm)':'rainfall','일강수량(mm)':'rainfall'})
    if not {'date','rainfall'}.issubset(r.columns): raise ValueError('강수 파일에는 날짜와 강수량(mm) 열이 필요합니다.')
    r['date']=r['date'].fillna('').str.strip();r=r[r['date']!=''].copy()
    r['date']=pd.to_datetime(r['date'],errors='raise').dt.normalize()
    if r['date'].duplicated().any(): raise ValueError('강수 파일은 한 지점의 하루 한 행이어야 합니다.')
    r['rainfall']=pd.to_numeric(r['rainfall'],errors='raise')
    if (r['rainfall'].dropna()<0).any():raise ValueError('음수 강수량을 확인하세요.')
    w=weather.copy()
    if 'date' not in w: raise ValueError('rn 일별 강수량은 일별 기온 자료와 결합해야 합니다.')
    w['date']=pd.to_datetime(w['date'],errors='raise').dt.normalize()
    if '지점' in w and '지점' in r:
        a=set(w['지점'].dropna().astype(str));b=set(r['지점'].dropna().astype(str))
        if a!=b:raise ValueError('기온과 강수의 관측 지점이 다릅니다.')
    if 'rainfall' not in w: w['rainfall']=float('nan')
    r=r.rename(columns={'rainfall':'rn_rainfall'})
    w=w.merge(r[['date','rn_rainfall']],on='date',how='left',validate='one_to_one')
    conflict=w['rainfall'].notna() & w['rn_rainfall'].notna() & ((w['rainfall']-w['rn_rainfall']).abs()>1e-8)
    if conflict.any():raise ValueError('기존 rainfall과 rn 파일 값이 충돌합니다. 같은 기간·지점의 파일인지 확인하세요.')
    w['rainfall']=w['rn_rainfall'].combine_first(w['rainfall'])
    return w.drop(columns='rn_rainfall'), f'강수 파일: {source.name} / 기온 날짜와 일치하는 강수 행: {w.date.isin(r.date).sum()}일'
