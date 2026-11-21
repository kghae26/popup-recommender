# -*- coding: utf-8 -*-
"""
팝업스토어 장소 추천 웹앱 (Streamlit)

실행:  streamlit run app.py
준비:  노트북 셀 R이 만든 artifacts/ 폴더를 이 파일과 같은 위치에 둡니다.
       artifacts/places.parquet, tag_matrix.npy, intens.npy, idf.npy, model.json
       (있으면 사용) age_index.parquet, top_of.json, cases.csv, eval.json

계산은 노트북 셀 O·P·Q와 같습니다.
  최종 점수 = 적합도(z) + 0.3 × 특색(z, ±3에서 자름)     ← 이 점수로 상위 후보를 고름
            + λ × 유동 점수 + 연령 가점                  ← 후보 안에서만 더함
"""
import json
import re
from collections import Counter
from pathlib import Path

import altair as alt
import pydeck as pdk
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='팝업 장소 추천', page_icon='📍', layout='wide')
ART = Path(__file__).parent / 'artifacts'

# ──────────────────────────────────────────────────────────────
# 0. 화면에 쓰는 이름과 색
#    축마다 색을 하나씩 정하고, 기여도가 클수록 진하게(투명도↓) 칠합니다.
# ──────────────────────────────────────────────────────────────
AX_KO = {'situation': '상황', 'material': '소재', 'value': '가치', 'issue': '장소 과제',
         'aesthetic': '분위기', 'audience': '타깃', 'scale': '공간 단위', 'season': '계절'}
AX_RGB = {'situation': (214, 69, 65), 'material': (41, 128, 185), 'value': (39, 174, 96),
          'issue': (127, 140, 141), 'aesthetic': (142, 68, 173), 'audience': (230, 126, 34),
          'scale': (22, 160, 133), 'season': (183, 149, 11)}
AGE_OPTIONS = ['10대', '20대', '30대', '40대', '50대', '60대', '70대 이상']
GOAL_KO = {'balanced': '균형 (방문자 + 체류)', 'awareness': '인지도 (방문자 + 검색 순위)', 'sales': '판매 (체류 + 야간)'}


def chip(label, axis, strength=1.0, title=''):
    """태그 하나를 둥근 칩으로 그림. strength(0~1)가 클수록 진한 색."""
    r, g, b = AX_RGB.get(axis, (120, 120, 120))
    a = 0.12 + 0.88 * float(np.clip(strength, 0, 1))
    fg = 'white' if a > 0.55 else '#222'
    return (f'<span title="{title}" style="background:rgba({r},{g},{b},{a:.2f});color:{fg};'
            f'border:1px solid rgba({r},{g},{b},.7);border-radius:12px;padding:2px 9px;margin:2px 3px;'
            f'display:inline-block;font-size:.85rem;white-space:nowrap">{label}</span>')


def legend_html():
    return ' '.join(chip(AX_KO[a], a, .75) for a in AX_KO)


# ──────────────────────────────────────────────────────────────
# 1. 산출물 불러오기 (한 번만 읽고 캐시)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='모델 파일을 불러오는 중...')
def load():
    md = json.load(open(ART / 'model.json', encoding='utf-8'))
    pl = pd.read_parquet(ART / 'places.parquet')
    if 'text' not in pl:                         # GitHub 25MB 제한 때문에 소개글을 text_0, text_1 … 파일로 나눠 저장한 경우
        parts = sorted(ART.glob('text_*.parquet'), key=lambda f: int(re.search(r'(\d+)', f.stem).group(1)))
        pl['text'] = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)['text'] if parts else ''
    tm = np.load(ART / 'tag_matrix.npy').astype(np.float32)
    intens = np.load(ART / 'intens.npy').astype(np.float32)
    idf = np.load(ART / 'idf.npy').astype(np.float32)
    tag_cols = md['tag_cols']
    tag_axis = np.array([c.split(':')[0] for c in tag_cols])

    # 자주 쓰는 배열을 미리 만들어 둠
    txt = pl['text'].fillna('').to_numpy().astype(str)          # np.char.find가 되려면 문자열 배열이어야 함
    seasons = [set(str(s).split('|')) if isinstance(s, str) and s else set() for s in pl.get('seasons', pd.Series([''] * len(pl)))]
    fm1 = pd.to_numeric(pl.get('fest_m1', np.nan), errors='coerce').to_numpy(float) if 'fest_m1' in pl else np.full(len(pl), np.nan)
    fm2 = pd.to_numeric(pl['fest_m2'], errors='coerce').to_numpy(float) if 'fest_m2' in pl else fm1.copy()
    vis_raw = pd.to_numeric(pl.get('vis_raw', 0), errors='coerce').to_numpy(float) if 'vis_raw' in pl else np.zeros(len(pl))
    name_base = pl['kto_title'].fillna('').str.replace(r'\s*\(.*?\)', '', regex=True).str.strip()
    group = pl['_group'].fillna('기타').to_numpy() if '_group' in pl else np.array(['기타'] * len(pl))
    sido = pl['_dl_sido'].fillna('?').to_numpy() if '_dl_sido' in pl else np.array(['?'] * len(pl))
    sgg = pl['_dl_sgg'].fillna('?').to_numpy() if '_dl_sgg' in pl else np.array(['?'] * len(pl))

    # 비슷한 장소 찾기용: 장소별 태그 벡터를 길이 1로 정규화 (코사인 유사도 = 내적)
    tmn = tm / (np.linalg.norm(tm, axis=1, keepdims=True) + 1e-9)

    # 연령 지수 (없으면 연령 가점은 0)
    age = pd.read_parquet(ART / 'age_index.parquet') if (ART / 'age_index.parquet').exists() else pd.DataFrame()
    top_of = json.load(open(ART / 'top_of.json', encoding='utf-8')) if (ART / 'top_of.json').exists() else {}
    cases = pd.read_csv(ART / 'cases.csv') if (ART / 'cases.csv').exists() else pd.DataFrame()
    ev = json.load(open(ART / 'eval.json', encoding='utf-8')) if (ART / 'eval.json').exists() else {}
    return dict(md=md, pl=pl, tm=tm, intens=intens, idf=idf, tag_cols=tag_cols, tag_axis=tag_axis, txt=txt,
                seasons=seasons, fm1=fm1, fm2=fm2, vis_raw=vis_raw, name_base=name_base, group=group,
                sido=sido, sgg=sgg, tmn=tmn, age=age, top_of=top_of, cases=cases, ev=ev)


if not (ART / 'model.json').exists():
    st.error(f'`{ART}` 폴더에 산출물이 없습니다. 노트북 셀 R을 실행해 만든 artifacts 폴더를 app.py 옆에 복사해 주세요.')
    st.stop()
A = load()
MD, PL, TAG_COLS = A['md'], A['pl'], A['tag_cols']
DEF = MD.get('defaults', {})
N = len(PL)


def zs(x):
    x = np.asarray(x, float)
    s = x.std() or 1.0
    return (x - x.mean()) / s


# ──────────────────────────────────────────────────────────────
# 2. 제품 설명 → 태그 (셀 C의 PRODUCT_RULES 그대로)
#    어떤 단어가 어떤 태그를 만들었는지도 함께 돌려줘서 화면에서 설명합니다.
# ──────────────────────────────────────────────────────────────
def product_tags(text):
    tags, fired = {}, []
    for pat, tg in MD['product_rules']:
        m = re.search(pat, text)
        if not m:
            continue
        fired.append({'word': m.group(0), 'pattern': pat, 'tags': tg})
        for k, v in tg.items():
            if k in TAG_COLS:
                tags[k] = max(tags.get(k, 0), v)
    return tags, fired


def weight_vector(tags):
    w = np.zeros(len(TAG_COLS), np.float32)
    for k, v in tags.items():
        w[TAG_COLS.index(k)] = v
    return w


# ──────────────────────────────────────────────────────────────
# 3. 점수 계산 (셀 D의 features_all, 셀 O의 distinct_score, 셀 Q의 age_bonus)
# ──────────────────────────────────────────────────────────────
def axis_features(w, month):
    """축별 일치 점수: 제품 태그 가중치로 가중평균한 장소의 태그 강도"""
    F = {}
    for ax, cols in MD['ax_cols'].items():
        d = w[cols].sum()
        F[ax] = (A['tm'][:, cols] * w[cols]).sum(1) / d if d > 0 else np.zeros(N, np.float32)
    if month:
        s = MD['season_of'][str(month)]
        F['season'] = np.array([1.0 if s in ss else 0.0 for ss in A['seasons']], np.float32)
        F['season'] = np.where(A['fm1'] == month, 1.0, F['season'])
    else:
        F['season'] = np.zeros(N, np.float32)
    return F


STOP = set(MD.get('stop_words', []))


def distinct_score(w, text):
    """특색 = 드문 태그(IDF) × 소개글에 많이 나옴(강도) + 제품 설명 단어가 소개글에 직접 나오는지"""
    tag_part = (A['intens'] * w * A['idf']).sum(1) / (w.sum() or 1)
    words = [x for x in re.findall(r'[가-힣A-Za-z]{2,}', text) if x not in STOP]
    words = list(dict.fromkeys(re.sub(r'(을|를|이|가|은|는|와|과|의|으로|로|에서)$', '', x) for x in words))
    direct = np.zeros(N, np.float32)
    for x in [x for x in words if len(x) >= 2]:
        hit = np.char.find(A['txt'], x) >= 0
        if 0 < hit.mean() < 0.2:                       # 장소의 20% 이상에 나오는 흔한 단어는 무시
            direct += hit * np.log(1 / hit.mean())
    return tag_part + 0.5 * direct


def age_bonus(target_age, w_age, w_top):
    """타깃 연령 가점: 그 지역 전 연령 평균보다 타깃 연령이 얼마나 더 많이 오나(쏠림)"""
    b = np.zeros(N, np.float32)
    if not target_age:
        return b
    bands = {re.match(r'(\d+)대', t).group(1) for t in target_age}
    age = A['age']
    if len(age):
        val = age.set_index(['sido', 'sgg'])
        skews = []
        for grp in ['다양성_관광객', '다양성_소비']:
            allc = [c for c in val.columns if c.startswith(grp) and re.search(r'(\d+)대', c)]
            tc = [c for c in allc if re.search(r'(\d+)대', c).group(1) in bands]
            if tc:
                skews.append(val[tc].mean(axis=1) - val[allc].mean(axis=1))
        if skews:
            sk = pd.concat(skews, axis=1).mean(axis=1)
            sk = (sk - sk.mean()) / (sk.std() or 1)
            keys = pd.MultiIndex.from_arrays([A['sido'], A['sgg']])
            b += w_age * np.nan_to_num(sk.reindex(keys).to_numpy(float), nan=0.0).astype(np.float32)
    top = {tuple(x) for band in bands for x in A['top_of'].get(band, [])}
    if top:
        b += w_top * np.array([1.0 if k in top else 0.0 for k in zip(A['sido'], A['sgg'])], np.float32)
    return b


def recommend(text, month, ages, goal, lam, w_distinct, min_vis_q, per_sgg, n_groups, per_group, pool=300, keep_fest=True):
    """셀 O(연결성 우선) + 셀 P(유형별 묶기) + 셀 Q(연령 가점).
    결과표와 함께, 근거 화면에서 쓸 중간 계산값(ctx)을 돌려줍니다."""
    tags, fired = product_tags(text)
    w = weight_vector(tags)
    F = axis_features(w, month)
    beta = {k: v for k, v in MD['beta_fit'].items() if k in F}
    fit = sum(b * F[k] for k, b in beta.items()) if beta else np.zeros(N)
    zfit = zs(fit)
    cap = DEF.get('distinct_cap', 3.0)
    zd = np.clip(zs(distinct_score(w, text)), -cap, cap)
    fit_score = zfit + w_distinct * zd

    crowd = np.zeros(N)
    for k, v in MD['crowd'][goal].items():
        if k in PL:
            crowd = crowd + v * pd.to_numeric(PL[k], errors='coerce').fillna(0).to_numpy(float)

    # 거르기: 팝업 달에 안 열리는 축제 / 방문자 하위 q% 시군구 / 산·봉우리처럼 설치가 어려운 이름
    if month:
        fm1, fm2 = A['fm1'], np.where(np.isnan(A['fm2']), A['fm1'], A['fm2'])
        held = np.where(fm1 <= fm2, (fm1 <= month) & (month <= fm2),       # 12월~2월처럼 해를 넘기는 축제도 처리
                        (month >= fm1) | (month <= fm2))
        bad_fest = ~np.isnan(fm1) & ~held
        fit_score = np.where(bad_fest, -np.inf, fit_score)
    vis = np.nan_to_num(A['vis_raw'], nan=0)
    ok = vis >= (np.nanquantile(A['vis_raw'], min_vis_q) if np.isfinite(A['vis_raw']).any() else 0)
    if DEF.get('drop_name'):
        pat = re.sub(r'\((?!\?)', '(?:', DEF['drop_name'])        # 괄호를 비캡처 그룹으로 (pandas 경고 방지)
        ok &= ~A['name_base'].str.contains(pat, regex=True).to_numpy()
    fit_score = np.where(ok, fit_score, -np.inf)

    bonus = age_bonus(ages, DEF.get('w_age', .3), DEF.get('w_top', .2))
    cand = np.argsort(-fit_score)[:pool]
    cand = cand[np.isfinite(fit_score[cand])]
    final = np.full(N, -np.inf)
    final[cand] = fit_score[cand] + lam * crowd[cand] + bonus[cand]   # 후보 안에서만 유동·연령 반영

    picked, cnt = [], Counter()
    for i in cand[np.argsort(-final[cand])]:
        key = (A['sido'][i], A['sgg'][i])
        if cnt[key] >= per_sgg:                         # 한 시군구가 목록을 독차지하지 않게
            continue
        cnt[key] += 1
        picked.append(int(i))
    big = pd.DataFrame({'_i': picked, '전체 순위': range(1, len(picked) + 1)})
    big['유형'] = A['group'][big['_i']] if len(big) else []
    if (big['유형'] != '기타').any():
        big = big[big['유형'] != '기타']
    order = list(big.groupby('유형', sort=False)['전체 순위'].min().sort_values().index[:n_groups])

    # 축제·행사는 이 프로젝트의 핵심 유형이라, 상위 유형 안에 못 들어도 항상 한 칸을 보여줌
    # (축제는 소개글이 짧아 적합도가 낮게 나오기 쉬움 → 후보 밖의 축제도 적합도 순으로 가져옴)
    FEST = '축제·행사'
    if keep_fest and FEST not in order:
        fest_i = np.where((A['group'] == FEST) & np.isfinite(fit_score))[0]
        if len(fest_i):
            fest_i = fest_i[np.argsort(-fit_score[fest_i])][:per_group]
            final[fest_i] = np.where(np.isfinite(final[fest_i]), final[fest_i],
                                     fit_score[fest_i] + lam * crowd[fest_i] + bonus[fest_i])
            extra = pd.DataFrame({'_i': fest_i.astype(int), '유형': FEST,
                                  '전체 순위': [int((fit_score > fit_score[j]).sum()) + 1 for j in fest_i]})
            big = pd.concat([big[big['유형'] != FEST], extra], ignore_index=True)
            order = order[:max(n_groups - 1, 1)] + [FEST]
    res = pd.concat([big[big['유형'] == g].sort_values('전체 순위').head(per_group).assign(유형순위=k) for k, g in enumerate(order, 1)],
                    ignore_index=True) if len(order) else big.head(0)
    ctx = dict(text=text, month=month, tags=tags, fired=fired, w=w, F=F, beta=beta, fit=fit, zfit=zfit, zd=zd,
               crowd=crowd, bonus=bonus, fit_score=fit_score, final=final, lam=lam, w_distinct=w_distinct, ok=ok)
    return res, ctx


# ──────────────────────────────────────────────────────────────
# 4. 근거 계산: 선택한 장소의 점수가 어디서 왔나
# ──────────────────────────────────────────────────────────────
def score_breakdown(i, ctx):
    """최종 점수를 축별 기여 + 특색 + 유동 + 연령으로 나눔.
    zs(fit)는 선형이라, 축 k의 기여 = β_k × (F_k[i] − 평균 F_k) / sd(fit) 로 정확히 나뉩니다."""
    sd = ctx['fit'].std() or 1.0
    rows = [{'항목': AX_KO.get(k, k), '축': k, '기여': b * (ctx['F'][k][i] - ctx['F'][k].mean()) / sd}
            for k, b in ctx['beta'].items()]
    rows += [{'항목': f'특색 ×{ctx["w_distinct"]}', '축': 'distinct', '기여': ctx['w_distinct'] * ctx['zd'][i]},
             {'항목': f'유동 ×{ctx["lam"]}', '축': 'crowd', '기여': ctx['lam'] * ctx['crowd'][i]},
             {'항목': '연령 가점', '축': 'age', '기여': float(ctx['bonus'][i])}]
    return pd.DataFrame(rows)


def tag_contrib(i, ctx):
    """태그별 기여: 제품 가중치 × 장소 태그 강도 × 축 가중치(β) / 축 가중치 합"""
    w, tm = ctx['w'], A['tm']
    out = []
    for ax, cols in MD['ax_cols'].items():
        d = w[cols].sum()
        if d <= 0:
            continue
        for j in cols:
            if w[j] <= 0:
                continue
            fitc = ctx['beta'].get(ax, 0) * tm[i, j] * w[j] / d
            out.append({'축': AX_KO[ax], 'axis': ax, '태그': TAG_COLS[j].split(':')[1], '제품 가중': round(float(w[j]), 2),
                        '장소 강도': round(float(tm[i, j]), 2), '소개글 언급(log)': round(float(A['intens'][i, j]), 2),
                        '기여': float(fitc), '_j': j})
    return pd.DataFrame(out).sort_values('기여', ascending=False) if out else pd.DataFrame()


def evidence_snippets(i, tc, n=4, width=45):
    """장소 소개글에서 태그 단어가 나온 부분을 잘라 보여줌 (태그 패턴이 저장돼 있을 때)"""
    pats = MD.get('tag_patterns', {})
    text = A['txt'][i]
    out = []
    for _, r in tc[tc['기여'] > 0].head(n).iterrows():
        p = pats.get(TAG_COLS[int(r['_j'])])
        m = re.search(p, text) if p else None
        if m:
            s, e = max(0, m.start() - width), min(len(text), m.end() + width)
            out.append((r['axis'], r['태그'], text[s:m.start()], m.group(0), text[m.end():e]))
    return out


def similar_places(i, ctx, same_group=True, per_region=1):
    """선택한 장소와 태그 구성이 비슷한 장소를 시·도별로 (코사인 유사도)"""
    s = A['tmn'] @ A['tmn'][i]
    s[i] = -1
    s[~ctx['ok']] = -1                                  # 최소 규모 조건은 똑같이 적용
    s[A['name_base'].to_numpy() == A['name_base'].iloc[i]] = -1
    if same_group:
        s[A['group'] != A['group'][i]] = -1
    df = pd.DataFrame({'_i': np.arange(N), '시도': A['sido'], '시군구': A['sgg'], '유사도': s})
    df = df[(df['유사도'] > 0) & (df['시도'] != A['sido'][i])]
    df = df.sort_values('유사도', ascending=False).groupby('시도').head(per_region)
    common = []
    for j in df['_i']:
        both = A['tm'][i] * A['tm'][j]
        common.append(' '.join(chip(TAG_COLS[k].split(':')[1], A['tag_axis'][k], .55) for k in np.argsort(-both)[:3] if both[k] > 0))
    df['장소'] = PL['kto_title'].to_numpy()[df['_i']]
    df['유형'] = A['group'][df['_i']]
    df['공통 태그'] = common
    df['제품 적합 순위'] = [int((ctx['fit_score'] > ctx['fit_score'][j]).sum()) + 1 for j in df['_i']]
    return df


def photo_url(i):
    for c in ['kto_firstimage', 'firstimage']:
        if c in PL and isinstance(PL.at[i, c], str) and PL.at[i, c].startswith('http'):
            return PL.at[i, c].replace('http://', 'https://')
    return None


def latlon(idx):
    df = pd.DataFrame({'lat': pd.to_numeric(PL.loc[idx, 'kto_mapy'], errors='coerce').to_numpy() if 'kto_mapy' in PL else np.nan,
                       'lon': pd.to_numeric(PL.loc[idx, 'kto_mapx'], errors='coerce').to_numpy() if 'kto_mapx' in PL else np.nan})
    return df.dropna()


# ──────────────────────────────────────────────────────────────
# 5. 화면: 사이드바(입력) → 제품 태그 → 유형별 추천(더보기) → 선택 장소 근거 → 비슷한 장소 → 두 장소 비교
# ──────────────────────────────────────────────────────────────
st.title('📍 팝업스토어 장소 추천')
st.caption('유명해서가 아니라 **제품과 연결되기 때문에** 추천합니다. 유명세는 최소 규모 거르기와 후보 안 순서 조정에만 씁니다.')

with st.sidebar:
    st.header('입력')
    with st.form('input'):
        text = st.text_area('제품·서비스 설명', '핸드메이드 입욕제·비누·바디케어 제품을 판다.', height=110,
                            help='제품이 무엇인지, 어떤 상황에서 쓰는지를 적을수록 태그가 잘 붙습니다.')
        month = st.selectbox('팝업 월', [None] + list(range(1, 13)), index=7,
                             format_func=lambda m: '정하지 않음' if m is None else f'{m}월')
        ages = st.multiselect('타깃 연령', AGE_OPTIONS, default=['20대', '30대'])
        goal = st.radio('목표', list(GOAL_KO), format_func=GOAL_KO.get, horizontal=False)
        with st.expander('세부 설정'):
            lam = st.slider('유동인구 반영 (λ)', 0.0, 1.0, float(DEF.get('lam', .2)), .05,
                            help='0이면 연결성·특색만. 키울수록 유명한 지역이 위로 올라옵니다.')
            w_distinct = st.slider('특색 가중치', 0.0, 1.0, float(DEF.get('w_distinct', .3)), .05)
            min_vis_q = st.slider('방문자 하위 몇 % 시군구 제외', 0.0, 0.8, float(DEF.get('min_vis_q', .3)), .05)
            per_sgg = st.number_input('한 시군구 최대 장소 수', 1, 10, int(DEF.get('per_sgg', 2)))
            n_groups = st.number_input('보여줄 유형 수', 1, 12, int(DEF.get('n_groups', 6)))
            keep_fest = st.checkbox('축제·행사 유형은 항상 보기', value=True,
                                    help='팝업 월에 열리는 축제 중 제품과 가장 잘 맞는 곳을 따로 한 칸 보여줍니다.')
            per_group = st.number_input('유형별 장소 수', 1, 10, int(DEF.get('per_group', 3)))
        go = st.form_submit_button('추천 받기', type='primary', width='stretch')
    st.markdown('**태그 색 (축)**<br>' + legend_html(), unsafe_allow_html=True)

# 버튼을 누르면 계산해서 세션에 저장 → 장소를 클릭해도 다시 계산하지 않음
if go or 'res' not in st.session_state:
    with st.spinner('장소 점수를 계산하는 중...'):
        res, ctx = recommend(text, month, ages, goal, lam, w_distinct, min_vis_q, int(per_sgg), int(n_groups), int(per_group), keep_fest=keep_fest)
    st.session_state.update(res=res, ctx=ctx, sel=int(res['_i'].iloc[0]) if len(res) else None)
res, ctx = st.session_state['res'], st.session_state['ctx']

# ── 5-1. 제품 설명에서 뽑은 태그 + 설명 보완 도우미 ──
st.subheader('① 제품 설명에서 읽은 태그')
if not ctx['tags']:
    st.warning('설명에서 태그를 하나도 못 찾았습니다. 아래 "설명 보완 도우미"의 단어를 참고해 설명을 고쳐 보세요.')
else:
    # 설명 문장 안에서 태그를 만든 단어를 그 축 색으로 칠해서 보여줌
    shown = ctx['text']
    for f in sorted(ctx['fired'], key=lambda f: -len(f['word'])):
        ax = max(f['tags'], key=f['tags'].get).split(':')[0]
        r, g, b = AX_RGB.get(ax, (120, 120, 120))
        shown = shown.replace(f['word'], f'<b style="background:rgba({r},{g},{b},.25);border-radius:4px;padding:0 3px">{f["word"]}</b>', 1)
    st.markdown(f'> {shown}', unsafe_allow_html=True)
    chips = ' '.join(chip(f'{AX_KO[k.split(":")[0]]} · {k.split(":")[1]}', k.split(':')[0], v, f'가중치 {v}')
                     for k, v in sorted(ctx['tags'].items(), key=lambda x: (list(AX_KO).index(x[0].split(':')[0]), -x[1])))
    st.markdown(chips, unsafe_allow_html=True)
    st.caption('색이 진할수록 제품에서 강하게 읽힌 태그입니다. 태그는 장소 소개글의 태그와 비교되어 "축별 적합도"가 됩니다.')

with st.expander('✍️ 설명 보완 도우미: 이런 단어를 넣으면 이런 태그가 붙어요'):
    used_axes = {k.split(':')[0] for k in ctx['tags']}
    beta = MD['beta_fit']
    st.write('비어 있는 축 중 **선택을 잘 설명하는 축(가중치가 큰 축)**부터 보여줍니다. 실제 제품과 맞는 단어만 넣으세요.')
    rows = []
    for pat, tg in MD['product_rules']:
        if re.search(pat, ctx['text']):
            continue
        axes = {k.split(':')[0] for k in tg}
        rows.append({'예시 단어': ', '.join(pat.split('|')[:4]),
                     '붙는 태그': ' '.join(chip(k.split(':')[1], k.split(':')[0], v) for k, v in tg.items()),
                     '새 축': ', '.join(AX_KO[a] for a in axes - used_axes) or '-',
                     '_w': max(beta.get(a, 0) for a in axes) + (1 if axes - used_axes else 0)})
    rows = sorted(rows, key=lambda r: -r['_w'])[:12]
    st.markdown(pd.DataFrame(rows).drop(columns='_w').to_html(escape=False, index=False), unsafe_allow_html=True)

# ── 화면에서 함께 쓰는 도구: 커서를 올리면 이름이 뜨는 지도, 점수 계산, 근거 화면 ──
def deck_map(idx, labels=None, highlight=None, height=380, key=None):
    """pydeck 지도: 점에 커서를 올리면 장소 이름·주소가 뜸. labels를 주면 점 위에 번호를 씀."""
    idx = [int(j) for j in idx]
    df = pd.DataFrame({'_i': idx})
    df['lat'] = pd.to_numeric(PL.loc[idx, 'kto_mapy'], errors='coerce').to_numpy() if 'kto_mapy' in PL else np.nan
    df['lon'] = pd.to_numeric(PL.loc[idx, 'kto_mapx'], errors='coerce').to_numpy() if 'kto_mapx' in PL else np.nan
    df['name'] = PL['kto_title'].to_numpy()[idx]
    df['addr'] = PL['kto_addr1'].fillna('').to_numpy()[idx] if 'kto_addr1' in PL else ''
    df['group'] = A['group'][idx]
    df['label'] = [str(x) for x in labels] if labels is not None else ''
    hl = set(highlight) if isinstance(highlight, (list, tuple, set)) else {highlight}
    df['color'] = [[214, 69, 65, 230] if j in hl else [41, 128, 185, 210] for j in idx]
    df = df.dropna(subset=['lat', 'lon'])
    if df.empty:
        st.caption('좌표 정보가 없습니다.')
        return
    span = max(df['lat'].max() - df['lat'].min(), df['lon'].max() - df['lon'].min())
    zoom = 12 if span < .02 else 10 if span < .2 else 8 if span < 1 else 6.5 if span < 3 else 5.8
    layers = [pdk.Layer('ScatterplotLayer', df, get_position='[lon, lat]', get_fill_color='color', get_radius=300,
                        radius_min_pixels=9, radius_max_pixels=16, stroked=True, get_line_color=[255, 255, 255],
                        line_width_min_pixels=1.5, pickable=True)]
    if labels is not None:                                   # 점 위에 목록 번호 (숫자만 쓰므로 한글 폰트 문제 없음)
        layers.append(pdk.Layer('TextLayer', df, get_position='[lon, lat]', get_text='label', get_size=13,
                                get_color=[255, 255, 255], get_text_anchor='"middle"', get_alignment_baseline='"center"'))
    deck = pdk.Deck(layers=layers, map_provider='carto', map_style=pdk.map_styles.CARTO_LIGHT,   # 토큰 없이 쓰는 밝은 지도
                    initial_view_state=pdk.ViewState(latitude=df['lat'].mean(), longitude=df['lon'].mean(), zoom=zoom),
                    tooltip={'html': '<b>{name}</b><br/>{group}<br/><span style="font-size:11px">{addr}</span>',
                             'style': {'backgroundColor': '#222', 'color': 'white', 'fontSize': '13px'}})
    st.pydeck_chart(deck, height=height, key=key)


def score_of(i, ctx):
    """최종 점수. 후보(상위 300곳) 밖이면 같은 식으로 계산해서 보여주고, 거르기에 걸렸으면 None."""
    if np.isfinite(ctx['final'][i]):
        return float(ctx['final'][i]), '후보 안'
    if not np.isfinite(ctx['fit_score'][i]):
        return None, '거르기 조건(방문자 수·축제 기간·산 이름)에 걸려 추천에서 제외'
    return float(ctx['fit_score'][i] + ctx['lam'] * ctx['crowd'][i] + ctx['bonus'][i]), '후보(상위 300곳) 밖'


def fit_rank_of(i, ctx):
    return int((ctx['fit_score'] > ctx['fit_score'][i]).sum()) + 1 if np.isfinite(ctx['fit_score'][i]) else None


def type_ranking(g, ctx, per_sgg, n=50):
    """유형 g의 상위 n곳 (더보기용). 거르기를 통과한 같은 유형 장소를 최종 점수 식으로 정렬"""
    idx = np.where((A['group'] == g) & np.isfinite(ctx['fit_score']))[0]
    sc = ctx['fit_score'][idx] + ctx['lam'] * ctx['crowd'][idx] + ctx['bonus'][idx]
    out, cnt = [], Counter()
    for j in idx[np.argsort(-sc)]:
        key = (A['sido'][j], A['sgg'][j])
        if cnt[key] >= per_sgg:
            continue
        cnt[key] += 1
        out.append(int(j))
        if len(out) >= n:
            break
    return out


def top_chips(i, ctx, k=4):
    tc = tag_contrib(i, ctx)
    if not len(tc):
        return ''
    mx = tc['기여'].max() or 1
    return ' '.join(chip(t['태그'], t['axis'], t['기여'] / mx) for _, t in tc.head(k).iterrows() if t['기여'] > 0)


def breakdown_chart(bd):
    color_of = {AX_KO[k]: f'rgb{AX_RGB[k]}' for k in AX_KO}
    bd = bd.assign(색=[color_of.get(h, '#888888') for h in bd['항목']])
    chart = alt.Chart(bd).mark_bar().encode(
        x=alt.X('기여:Q', title='최종 점수에 대한 기여 (+는 올림, −는 내림)'),
        y=alt.Y('항목:N', sort=None, title=None),
        color=alt.Color('색:N', scale=None),
        tooltip=['항목', alt.Tooltip('기여:Q', format='.2f')])
    return chart + alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='gray').encode(x='x:Q')


def render_rationale(i, ctx, key):
    """한 장소의 순위 근거 화면 (사진·지표 / 점수 구성 그래프 / 태그 기여 / 소개글 속 근거)"""
    left, right = st.columns([1, 2])
    with left:
        url = photo_url(i)
        if url:
            st.image(url, width='stretch')
        else:
            st.caption('(사진 없음)')
        st.write(PL.at[i, 'kto_addr1'] if 'kto_addr1' in PL else '')
        sc, where = score_of(i, ctx)
        fr = fit_rank_of(i, ctx)
        c1, c2 = st.columns(2)
        c1.metric('적합 순위 (전체)', f'{fr:,}위' if fr else '-', help='유동·연령을 더하기 전, 연결성·특색만으로 매긴 순위')
        c2.metric('최종 점수', f'{sc:.2f}' if sc is not None else '-', help=where)
        c1.metric('적합도 (z)', f'{ctx["zfit"][i]:.2f}', help='0이 전체 평균, 1이면 표준편차 1개만큼 위')
        c2.metric('특색 (z)', f'{ctx["zd"][i]:.2f}', help='드문 태그로 맞을수록, 제품 단어가 소개글에 직접 나올수록 높음')
        if where != '후보 안':
            st.caption(f'※ {where}')
        if 'vis_raw' in PL and pd.notna(PL.at[i, 'vis_raw']):
            st.caption(f'시군구 연 방문자 약 {PL.at[i, "vis_raw"] / 1e4:,.0f}만 명'
                       + (' · 검색 Top100' if 'top100' in PL and PL.at[i, 'top100'] > 0 else ''))
    with right:
        st.markdown('**점수 구성**: 축별 막대의 합이 적합도(z)이고, 여기에 특색·유동·연령이 더해져 최종 점수가 됩니다.')
        st.altair_chart(breakdown_chart(score_breakdown(i, ctx)), width='stretch')
        tc = tag_contrib(i, ctx)
        if len(tc):
            mx = tc['기여'].max() or 1
            st.markdown('**태그별 기여** (진할수록 이 장소에서 점수를 많이 만든 태그)')
            st.markdown(' '.join(chip(f'{t["태그"]} {t["기여"]:.2f}', t['axis'], max(t['기여'], 0) / mx,
                                      f'제품 {t["제품 가중"]} × 장소 {t["장소 강도"]}') for _, t in tc.iterrows()),
                        unsafe_allow_html=True)
            st.dataframe(tc.drop(columns=['axis', '_j']), hide_index=True, width='stretch', key=f'{key}_tc',
                         column_config={'기여': st.column_config.ProgressColumn('기여', min_value=0, max_value=float(mx), format='%.2f')})
        snip = evidence_snippets(i, tc) if len(tc) else []
        if snip:
            st.markdown('**소개글 속 근거**')
            for ax, tag, a, m, b in snip:
                r, g, bb = AX_RGB[ax]
                st.markdown(f'{chip(tag, ax, .8)} …{a}<b style="background:rgba({r},{g},{bb},.3)">{m}</b>{b}…', unsafe_allow_html=True)
        with st.expander('장소 소개글 전체'):
            st.write(A['txt'][i][:3000])


def place_label(j):
    return f'{PL.at[j, "kto_title"]} · {A["sido"][j]} {A["sgg"][j]} · {A["group"][j]}'


def place_picker(label, default, key):
    """검색창에 이름 일부를 치면 맞는 장소를 골라 넣을 수 있음 (비워 두면 기본 장소)"""
    q = st.text_input(f'{label} 검색', key=f'{key}_q', placeholder='장소 이름 일부 (예: 해운대, 온천)')
    if q.strip():
        hit = np.where(PL['kto_title'].fillna('').str.contains(q.strip(), case=False, regex=False).to_numpy())[0]
        if not len(hit):
            st.caption('검색 결과가 없습니다. 기본 장소를 씁니다.')
            return default
        hit = hit[np.argsort(-np.nan_to_num(ctx['fit_score'][hit], neginf=-1e9))][:50]   # 지금 제품에 잘 맞는 순으로
        return int(st.selectbox(f'{label} 선택 ({len(hit)}곳)', hit.tolist(), format_func=place_label, key=f'{key}_sel'))
    return default


# ── 5-2. 유형별 추천 목록 (기본 3곳, 더보기를 켜면 상위 50곳 목록 + 지도) ──
st.subheader('② 유형별 추천')
if res.empty:
    st.info('조건에 맞는 장소가 없습니다. 세부 설정에서 제외 기준을 낮춰 보세요.')
    st.stop()

types = list(dict.fromkeys(res['유형']))
tabs = st.tabs([f'{k}. {g}' for k, g in enumerate(types, 1)])
for tab, g in zip(tabs, types):
    with tab:
        more = st.toggle('더보기: 이 유형 상위 50곳 목록', key=f'more_{g}')
        if not more:
            # 기본 화면: 사진 카드 3장
            sub = res[res['유형'] == g]
            cols = st.columns(len(sub))
            for col, (_, r) in zip(cols, sub.iterrows()):
                i = int(r['_i'])
                with col, st.container(border=True):
                    url = photo_url(i)
                    if url:
                        st.image(url, width='stretch')
                    st.markdown(f'**{PL.at[i, "kto_title"]}**')
                    st.caption(f'{PL.at[i, "kto_addr1"] if "kto_addr1" in PL else ""} · 전체 {r["전체 순위"]}위')
                    st.markdown(top_chips(i, ctx), unsafe_allow_html=True)
                    if st.button('근거 보기', key=f'sel_{g}_{i}', width='stretch',
                                 type='primary' if st.session_state.get('sel') == i else 'secondary'):
                        st.session_state['sel'] = i
                        st.rerun()
        else:
            # 더보기: 왼쪽 간단 목록(10곳씩), 오른쪽 그 10곳의 지도
            ranked = type_ranking(g, ctx, int(per_sgg))
            pages = max(1, -(-len(ranked) // 10))
            page = st.segmented_control('페이지', [f'{p * 10 + 1}~{min((p + 1) * 10, len(ranked))}위' for p in range(pages)],
                                        default=f'1~{min(10, len(ranked))}위', key=f'page_{g}') or f'1~{min(10, len(ranked))}위'
            start = int(page.split('~')[0]) - 1
            shown = ranked[start:start + 10]
            lc, rc = st.columns([3, 2])
            with lc:
                for n, j in enumerate(shown, start + 1):
                    with st.container(border=True):
                        a, b = st.columns([5, 1])
                        a.markdown(f'**{n}. {PL.at[j, "kto_title"]}** <span style="color:gray;font-size:.8rem">'
                                   f'{A["sido"][j]} {A["sgg"][j]}</span><br>{top_chips(j, ctx, 3)}', unsafe_allow_html=True)
                        if b.button('근거', key=f'more_sel_{g}_{j}', type='primary' if st.session_state.get('sel') == j else 'secondary'):
                            st.session_state['sel'] = j
                            st.rerun()
            with rc:
                deck_map(shown, labels=list(range(start + 1, start + 1 + len(shown))),
                         highlight=st.session_state.get('sel'), height=640, key=f'map_more_{g}_{start}')
                st.caption('점에 커서를 올리면 장소 이름이 나옵니다. 빨간 점은 지금 근거를 보고 있는 장소입니다.')

with st.expander('🗺️ 추천 장소 전체 지도'):
    deck_map(res['_i'], highlight=st.session_state.get('sel'), key='map_all')

# ── 5-3. 선택한 추천 장소의 순위 근거 ──
i = st.session_state.get('sel')
if i is None:
    st.stop()
st.divider()
st.subheader(f'③ 왜 「{PL.at[i, "kto_title"]}」인가')
render_rationale(i, ctx, key='main')

# ── 5-4. 이 장소와 비슷한 장소 (지역별) + 고른 장소의 근거 ──
st.divider()
st.subheader(f'④ 「{PL.at[i, "kto_title"]}」와 비슷한 장소를 다른 지역에서 찾기')
c1, c2 = st.columns(2)
same_group = c1.toggle('같은 유형만', value=True)
per_region = c2.number_input('시·도별 장소 수', 1, 5, 1)
sim = similar_places(i, ctx, same_group, int(per_region))
pick = None
if sim.empty:
    st.info('비슷한 장소를 찾지 못했습니다.')
else:
    st.caption('태그 구성이 얼마나 비슷한지(코사인 유사도)로 찾습니다. "제품 적합 순위"는 지금 입력한 제품 기준 전체 순위입니다.')
    lc, rc = st.columns([3, 2])
    with lc:
        st.markdown(sim[['시도', '시군구', '장소', '유형', '유사도', '제품 적합 순위', '공통 태그']]
                    .assign(유사도=lambda d: d['유사도'].round(2)).to_html(escape=False, index=False), unsafe_allow_html=True)
    with rc:
        pick = int(st.selectbox('자세히 볼 장소', sim['_i'].tolist(), format_func=place_label, key='sim_pick'))
        deck_map([i] + sim['_i'].tolist(), highlight=[i, pick], height=420, key='map_sim')
        st.caption('빨간 점: 기준 장소와 지금 고른 장소. 점에 커서를 올리면 이름이 나옵니다.')
    # 고른 비슷한 장소의 근거는 ③과 따로, 이 아래에 보여줌
    with st.container(border=True):
        st.markdown(f'#### 🔎 「{PL.at[pick, "kto_title"]}」의 근거')
        render_rationale(pick, ctx, key='sim')

# ── 5-5. 두 장소 근거 비교 (검색으로 아무 장소나 넣을 수 있음) ──
st.divider()
st.subheader('⑤ 두 장소 근거 비교')
st.caption('기본은 ③의 추천 장소와 ④에서 고른 장소입니다. 검색창에 이름을 치면 다른 장소로 바꿀 수 있습니다.')
pa, pb = st.columns(2)
with pa:
    ia = place_picker('장소 A', i, 'cmp_a')
with pb:
    ib = place_picker('장소 B', pick if pick is not None else (int(res['_i'].iloc[1]) if len(res) > 1 else i), 'cmp_b')

CMP_COLOR = {'A': '#2c3e50', 'B': '#e67e22'}
ca, cb = st.columns(2)
for col, j, lab in [(ca, ia, 'A'), (cb, ib, 'B')]:
    with col, st.container(border=True):
        st.markdown(f'<span style="color:{CMP_COLOR[lab]};font-weight:700">장소 {lab}</span> · **{PL.at[j, "kto_title"]}**',
                    unsafe_allow_html=True)
        st.caption(f'{PL.at[j, "kto_addr1"] if "kto_addr1" in PL else ""} · {A["group"][j]}')
        url = photo_url(j)
        if url:
            st.image(url, width='stretch')
        sc, where = score_of(j, ctx)
        fr = fit_rank_of(j, ctx)
        m1, m2, m3 = st.columns(3)
        m1.metric('적합 순위', f'{fr:,}위' if fr else '-')
        m2.metric('최종 점수', f'{sc:.2f}' if sc is not None else '-', help=where)
        m3.metric('특색 (z)', f'{ctx["zd"][j]:.2f}')
        st.markdown(top_chips(j, ctx, 6), unsafe_allow_html=True)

# 점수 구성을 두 장소 나란히 (막대 색 = 장소)
bd = pd.concat([score_breakdown(ia, ctx).assign(장소='A'), score_breakdown(ib, ctx).assign(장소='B')], ignore_index=True)
cmp_chart = alt.Chart(bd).mark_bar().encode(
    x=alt.X('기여:Q', title='최종 점수에 대한 기여'),
    y=alt.Y('항목:N', sort=None, title=None),
    yOffset='장소:N',
    color=alt.Color('장소:N', scale=alt.Scale(domain=['A', 'B'], range=[CMP_COLOR['A'], CMP_COLOR['B']])),
    tooltip=['장소', '항목', alt.Tooltip('기여:Q', format='.2f')])
lc, rc = st.columns([3, 2])
with lc:
    st.markdown('**점수 구성 비교**: 막대가 길게 차이 나는 항목이 두 장소의 순위를 가른 이유입니다.')
    st.altair_chart(cmp_chart + alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='gray').encode(x='x:Q'), width='stretch')
with rc:
    deck_map([ia, ib], labels=['A', 'B'], highlight=[ia], height=360, key='map_cmp')

# 태그별 기여 비교표: 어느 태그에서 차이가 났나
ta, tb = tag_contrib(ia, ctx), tag_contrib(ib, ctx)
if len(ta) and len(tb):
    t = ta[['축', '태그', '기여']].merge(tb[['축', '태그', '기여']], on=['축', '태그'], how='outer', suffixes=(' A', ' B')).fillna(0)
    t['차이 (A−B)'] = t['기여 A'] - t['기여 B']
    t = t.sort_values('차이 (A−B)', key=abs, ascending=False)
    mx = float(max(t['기여 A'].max(), t['기여 B'].max(), 1e-9))
    st.markdown('**태그별 기여 비교** (차이가 큰 태그부터)')
    st.dataframe(t, hide_index=True, width='stretch', key='cmp_tags',
                 column_config={'기여 A': st.column_config.ProgressColumn('기여 A', min_value=0, max_value=mx, format='%.2f'),
                                '기여 B': st.column_config.ProgressColumn('기여 B', min_value=0, max_value=mx, format='%.2f'),
                                '차이 (A−B)': st.column_config.NumberColumn(format='%+.2f')})

# ── 5-6. 모델 설명 ──
with st.expander('ℹ️ 이 추천은 어떻게 만들어졌나'):
    st.markdown(f'''
- 장소 {N:,}곳의 소개글에서 키워드로 태그를 뽑고, 제품 설명에서도 같은 태그를 뽑아 **축별 적합도**를 계산합니다.
- 축별 가중치는 실제 장소연계 마케팅 사례로 추정한 조건부 로지스틱 회귀 계수입니다(유명세를 통제한 모형).
- 유명세(방문자·검색 순위)는 점수의 중심이 아니라, 최소 규모 거르기와 상위 후보 안의 순서 조정(λ)에만 씁니다.
- 한계: 사례 수가 많지 않아 가중치의 크기보다 **방향**을 믿는 것이 안전하고, 정확히 한 곳을 맞히기보다 **비슷한 유형의 후보를 넓게 보는 용도**에 맞습니다.
''')
    st.write('적합도 가중치:', pd.Series(MD['beta_fit']).rename(index=AX_KO).round(3).to_frame('가중치').T)
    if len(A['cases']):
        st.write(f'학습에 쓴 사례 {len(A["cases"])}건')
        st.dataframe(A['cases'].drop(columns=['pt_idx'], errors='ignore'), hide_index=True, width='stretch')
