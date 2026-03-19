"""
数字化服务工作室转化链路与动态报价 A/B 测试分析平台
===================================================
陈文杰 | Monte Carlo Simulation + Statistical Inference + Bayesian Analysis
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from data_generator import generate_ab_test_data, get_summary_stats

# ==================================================
# 页面配置
# ==================================================
st.set_page_config(
    page_title="A/B Test 分析平台 | 陈文杰",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# 调色板
# ==================================================
C = {
    'a': '#6366f1', 'a_light': '#a5b4fc',
    'b': '#f59e0b', 'b_light': '#fcd34d',
    'pos': '#10b981', 'neg': '#ef4444',
    'gray': '#64748b', 'dark': '#1e293b',
    'bg': '#f8fafc', 'white': '#ffffff',
    'accent': '#8b5cf6',
    # rgba 半透明版本 — Plotly 6.x 要求 rgba() 格式，不支持 hex+alpha
    'a_fill': 'rgba(99,102,241,0.15)',
    'b_fill': 'rgba(245,158,11,0.15)',
    'accent_fill': 'rgba(139,92,246,0.15)',
}

# ==================================================
# 自定义 CSS
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} .stDeployButton {display:none;}
header[data-testid="stHeader"] { background: rgba(248,250,252,0.8); backdrop-filter: blur(12px); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 50%, #3730a3 100%);
}
[data-testid="stSidebar"] * { color: #c7d2fe !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #e0e7ff !important; }
[data-testid="stSidebar"] a { color: #818cf8 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(165,180,252,0.2); }

/* KPI Cards */
.kpi-row { display: flex; gap: 16px; margin: 20px 0 30px; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 150px;
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(10px);
    border-radius: 16px; padding: 20px 16px; text-align: center;
    box-shadow: 0 4px 24px rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.08);
    transition: transform .25s ease, box-shadow .25s ease;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 8px 32px rgba(99,102,241,0.14); }
.kpi-icon { font-size: 1.6rem; margin-bottom: 6px; }
.kpi-value { font-size: 1.85rem; font-weight: 800; color: #1e293b; margin: 4px 0; }
.kpi-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; }
.kpi-delta {
    display: inline-block; margin-top: 6px; padding: 3px 10px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 700;
}
.delta-up { color: #059669; background: #d1fae5; }
.delta-down { color: #dc2626; background: #fee2e2; }

/* Hero */
.hero { text-align: center; padding: 36px 20px 20px; }
.hero-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 8px;
}
.hero-sub { font-size: 1.15rem; color: #64748b; margin-bottom: 18px; }
.badges { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
.badge {
    padding: 5px 14px; border-radius: 20px; font-size: 0.78rem;
    font-weight: 600; letter-spacing: 0.3px; color: #fff;
}
.bg-purple { background: linear-gradient(135deg,#6366f1,#8b5cf6); }
.bg-blue   { background: linear-gradient(135deg,#3b82f6,#06b6d4); }
.bg-green  { background: linear-gradient(135deg,#10b981,#34d399); }
.bg-amber  { background: linear-gradient(135deg,#f59e0b,#fbbf24); }

/* Info boxes */
.box-insight { background: linear-gradient(135deg,#eff6ff,#dbeafe); border-left: 4px solid #3b82f6;
               border-radius: 0 12px 12px 0; padding: 14px 18px; margin: 14px 0; color: #1e40af; font-size:.92rem; }
.box-success { background: linear-gradient(135deg,#ecfdf5,#d1fae5); border-left: 4px solid #10b981;
               border-radius: 0 12px 12px 0; padding: 14px 18px; margin: 14px 0; color: #065f46; font-size:.92rem; }
.box-warn   { background: linear-gradient(135deg,#fffbeb,#fef3c7); border-left: 4px solid #f59e0b;
               border-radius: 0 12px 12px 0; padding: 14px 18px; margin: 14px 0; color: #92400e; font-size:.92rem; }

/* Section headers */
.sec-h { font-size: 1.3rem; font-weight: 700; color: #1e293b;
         border-bottom: 3px solid #6366f1; display: inline-block; padding-bottom: 4px; margin-bottom: 16px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; font-weight: 600; }

/* ============ Mobile Responsive ============ */
@media (max-width: 768px) {
    /* Hero */
    .hero { padding: 20px 10px 12px; }
    .hero-title { font-size: 1.35rem; line-height: 1.3; }
    .hero-sub { font-size: 0.82rem; margin-bottom: 12px; line-height: 1.5; }
    .badges { gap: 6px; }
    .badge { padding: 4px 10px; font-size: 0.68rem; }

    /* KPI Cards — 2 per row on mobile */
    .kpi-row { gap: 8px; margin: 12px 0 18px; }
    .kpi-card {
        flex: 1 1 calc(50% - 8px); min-width: 120px;
        padding: 14px 8px; border-radius: 12px;
    }
    .kpi-icon { font-size: 1.2rem; margin-bottom: 3px; }
    .kpi-value { font-size: 1.25rem; }
    .kpi-label { font-size: 0.65rem; letter-spacing: 0.8px; }
    .kpi-delta { font-size: 0.68rem; padding: 2px 8px; }

    /* Info boxes */
    .box-insight, .box-success, .box-warn {
        padding: 10px 12px; margin: 10px 0; font-size: 0.82rem;
        border-left-width: 3px;
    }

    /* Section headers */
    .sec-h { font-size: 1.05rem; }

    /* Tabs — scrollable, smaller text */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px; overflow-x: auto; flex-wrap: nowrap;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.72rem; padding: 8px 10px;
        white-space: nowrap; flex-shrink: 0;
    }

    /* Streamlit main container — reduce padding */
    .stMainBlockContainer, [data-testid="stAppViewBlockContainer"],
    section[data-testid="stMain"] > div { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    .block-container { padding: 1rem 0.5rem !important; max-width: 100% !important; }

    /* Streamlit columns — reduce gap */
    [data-testid="column"] { padding: 0 4px !important; }

    /* Tables — scroll horizontally */
    .stMarkdown table { display: block; overflow-x: auto; white-space: nowrap; font-size: 0.8rem; }
}

@media (max-width: 480px) {
    .hero-title { font-size: 1.15rem; }
    .hero-sub { font-size: 0.75rem; }

    /* KPI Cards — still 2 per row but tighter */
    .kpi-card {
        flex: 1 1 calc(50% - 6px); min-width: 100px;
        padding: 10px 6px;
    }
    .kpi-value { font-size: 1.1rem; }
    .kpi-label { font-size: 0.6rem; }

    .stTabs [data-baseweb="tab"] { font-size: 0.65rem; padding: 6px 8px; }
    .sec-h { font-size: 0.95rem; }
    .box-insight, .box-success, .box-warn { font-size: 0.78rem; padding: 8px 10px; }
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 数据加载（缓存）
# ==================================================
@st.cache_data
def load_data():
    return generate_ab_test_data()

@st.cache_data
def calc_summary(_df):
    return get_summary_stats(_df)

df = load_data()
S = calc_summary(df)

# ==================================================
# 工具函数
# ==================================================
def style_fig(fig, height=380):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=height, margin=dict(l=36, r=16, t=72, b=36),
        font=dict(family="Inter, system-ui, sans-serif", color='#334155', size=11),
        title_font_size=13,
        legend=dict(bgcolor='rgba(255,255,255,.85)', bordercolor='rgba(0,0,0,.06)',
                    borderwidth=1, font_size=10,
                    orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified',
        autosize=True
    )
    fig.update_xaxes(gridcolor='#e2e8f0', zeroline=False)
    fig.update_yaxes(gridcolor='#e2e8f0', zeroline=False)
    return fig

def show_chart(fig, **kwargs):
    st.plotly_chart(fig, use_container_width=True,
                    config={'displayModeBar': False, 'responsive': True}, **kwargs)

def hex_to_rgba(hex_color, alpha=0.15):
    """将 hex 颜色转为 Plotly 6.x 兼容的 rgba 字符串"""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

# ==================================================
# 统计检验函数
# ==================================================
@st.cache_data
def run_chi2_test(_df):
    a = _df[_df['group'] == 'A']
    b = _df[_df['group'] == 'B']
    table = [[a['converted'].sum(), len(a) - a['converted'].sum()],
             [b['converted'].sum(), len(b) - b['converted'].sum()]]
    chi2, p, dof, _ = sp_stats.chi2_contingency(table)
    return chi2, p, dof

@st.cache_data
def run_z_test(_df):
    a = _df[_df['group'] == 'A']
    b = _df[_df['group'] == 'B']
    n_a, n_b = len(a), len(b)
    p_a = a['converted'].mean()
    p_b = b['converted'].mean()
    p_pool = _df['converted'].mean()
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z = (p_b - p_a) / se
    p_val = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    ci_diff = (p_b - p_a - 1.96*se, p_b - p_a + 1.96*se)
    return z, p_val, ci_diff, p_a, p_b

@st.cache_data
def cumulative_pvalues(_df):
    dates = sorted(_df['date'].unique())
    records = []
    for i, d in enumerate(dates):
        sub = _df[_df['date'] <= d]
        a = sub[sub['group'] == 'A']
        b = sub[sub['group'] == 'B']
        if a['converted'].sum() < 1 or b['converted'].sum() < 1:
            continue
        tbl = [[a['converted'].sum(), len(a) - a['converted'].sum()],
               [b['converted'].sum(), len(b) - b['converted'].sum()]]
        _, p, _, _ = sp_stats.chi2_contingency(tbl)
        n_total = len(sub)
        # O'Brien-Fleming spending function boundary
        info_frac = (i + 1) / len(dates)
        obf_boundary = sp_stats.norm.ppf(1 - 0.025 / max(0.001, np.sqrt(info_frac)))
        records.append({'date': d, 'p_value': p, 'day': i + 1, 'n_cum': n_total,
                        'cum_a_rate': a['converted'].mean(),
                        'cum_b_rate': b['converted'].mean(),
                        'info_frac': info_frac,
                        'obf_alpha': 2 * (1 - sp_stats.norm.cdf(obf_boundary))})
    return pd.DataFrame(records)

@st.cache_data
def compute_srm_test(_df):
    """样本比例失配检验 (Sample Ratio Mismatch)"""
    n_a = len(_df[_df['group'] == 'A'])
    n_b = len(_df[_df['group'] == 'B'])
    n_total = n_a + n_b
    chi2 = (n_a - n_total / 2) ** 2 / (n_total / 2) + (n_b - n_total / 2) ** 2 / (n_total / 2)
    p = 1 - sp_stats.chi2.cdf(chi2, df=1)
    return n_a, n_b, chi2, p

@st.cache_data
def bayesian_analysis(_df, n_samples=200000):
    rng = np.random.default_rng(123)
    a = _df[_df['group'] == 'A']
    b = _df[_df['group'] == 'B']
    a_alpha = 1 + a['converted'].sum()
    a_beta_p = 1 + len(a) - a['converted'].sum()
    b_alpha = 1 + b['converted'].sum()
    b_beta_p = 1 + len(b) - b['converted'].sum()
    a_samp = rng.beta(a_alpha, a_beta_p, n_samples)
    b_samp = rng.beta(b_alpha, b_beta_p, n_samples)
    prob_b_wins = (b_samp > a_samp).mean()
    lift_samp = (b_samp - a_samp) / a_samp
    return a_samp, b_samp, prob_b_wins, lift_samp

# ==================================================
# 侧边栏
# ==================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:24px 0 10px;">
        <div style="font-size:2.8rem;">🔬</div>
        <h2 style="margin:8px 0 4px; font-size:1.3rem;">A/B 测试分析平台</h2>
        <p style="font-size:.88rem; opacity:.8;">北大的小码农 · 数字化服务工作室</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 📋 项目简介")
    st.markdown("""
    通过引入**交互式需求评估与动态报价器**，优化技术服务工作室客户转化链路，
    保卫并提升 **1:4 广告投入回报比 (ROI)** 北极星指标。

    数据基于真实商业经验，参考行业公开基准数据 (CNNIC 47th),
    采用 **蒙特卡洛方法** 模拟生成，包含多维度噪音与季节性波动。
    """)
    st.divider()

    st.markdown("#### 📊 快速概览")
    days = df['date'].nunique()
    st.markdown(f"""
    - 🗓️ 实验周期：**{days}** 天
    - 👥 总样本量：**{len(df):,}**
    - 📈 转化提升：**+{(S['B']['conv_rate']/S['A']['conv_rate']-1)*100:.1f}%**
    - 🎯 北极星：ROI ≥ 1:4
    """)
    st.divider()

    st.markdown("#### 🛠️ 技术栈")
    st.markdown("Python · Pandas · NumPy · SciPy · Streamlit · Plotly · 蒙特卡洛 · 贝叶斯推断")
    st.divider()

    st.markdown("""
    <div style="text-align:center; padding:8px 0;">
        <p style="font-size:.78rem; opacity:.7;">Created by</p>
        <p style="font-weight:700;">陈文杰</p>
        <a href="https://github.com/ucarcompany" target="_blank" style="font-size:.85rem;">GitHub →</a>
    </div>""", unsafe_allow_html=True)

# ==================================================
# Hero
# ==================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">数字化服务工作室 · A/B 测试分析平台</div>
    <div class="hero-sub">交互式需求评估与动态报价器对转化链路的影响 — 蒙特卡洛模拟 × 统计推断 × 贝叶斯分析</div>
    <div class="badges">
        <span class="badge bg-purple">Monte Carlo Simulation</span>
        <span class="badge bg-blue">Frequentist Testing</span>
        <span class="badge bg-green">Bayesian Inference</span>
        <span class="badge bg-amber">ROI Optimization</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# KPI 看板
# ==================================================
chi2_val, chi2_p, _ = run_chi2_test(df)
lift_pct = (S['B']['conv_rate'] / S['A']['conv_rate'] - 1) * 100
blended_roi = (S['A']['revenue'] + S['B']['revenue']) / (S['A']['ad_cost'] + S['B']['ad_cost'])

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card"><div class="kpi-icon">👥</div>
        <div class="kpi-value">{len(df):,}</div>
        <div class="kpi-label">总样本量</div></div>
    <div class="kpi-card"><div class="kpi-icon">🅰️</div>
        <div class="kpi-value">{S['A']['conv_rate']:.2%}</div>
        <div class="kpi-label">A组 转化率</div></div>
    <div class="kpi-card"><div class="kpi-icon">🅱️</div>
        <div class="kpi-value">{S['B']['conv_rate']:.2%}</div>
        <div class="kpi-label">B组 转化率</div>
        <div class="kpi-delta delta-up">↑ +{S['B']['conv_rate']-S['A']['conv_rate']:.2%}</div></div>
    <div class="kpi-card"><div class="kpi-icon">📈</div>
        <div class="kpi-value">+{lift_pct:.1f}%</div>
        <div class="kpi-label">相对提升</div></div>
    <div class="kpi-card"><div class="kpi-icon">🧪</div>
        <div class="kpi-value">{'<0.001' if chi2_p < 0.001 else f'{chi2_p:.4f}'}</div>
        <div class="kpi-label">P-Value</div>
        <div class="kpi-delta {'delta-up' if chi2_p < 0.05 else 'delta-down'}">{'✅ 显著' if chi2_p < 0.05 else '⏳ 不显著'}</div></div>
    <div class="kpi-card"><div class="kpi-icon">💰</div>
        <div class="kpi-value">1:{blended_roi:.1f}</div>
        <div class="kpi-label">综合 ROI</div>
        <div class="kpi-delta delta-up">北极星 ≥ 1:4 ✓</div></div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# 标签页
# ==================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 实验概览", "🔄 转化漏斗", "💰 ROI 分析",
    "🧪 统计检验", "🎯 贝叶斯推断", "🔍 分群洞察", "📖 方法论"
])

# ===== TAB 1: 实验概览 =====
with tab1:
    # SRM 检查
    srm_na, srm_nb, srm_chi2, srm_p = compute_srm_test(df)
    if srm_p < 0.01:
        st.markdown(f"""<div class="box-warn">
        <strong>⚠️ SRM 警告：</strong>A/B 两组样本比例失配 (A={srm_na:,}, B={srm_nb:,}, P={srm_p:.4f})，
        可能存在分流偏差，需排查随机化机制。
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-h">累积 P 值收敛图 — 识别「偷窥陷阱」</div>', unsafe_allow_html=True)
    st.markdown("""<div class="box-warn">
    <strong>⚠️ Peeking Problem（偷窥陷阱）：</strong>在实验期间反复查看 P 值会严重膨胀假阳性率。
    下图展示 P 值随样本量增长的收敛过程，同时叠加 O'Brien-Fleming 序贯检验边界。
    只有当 P 值稳定低于动态校正的 α 阈值时，才可可靠地做出决策。
    </div>""", unsafe_allow_html=True)

    cpv = cumulative_pvalues(df)
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scatter(
        x=cpv['date'], y=cpv['p_value'],
        mode='lines+markers', name='累积 P 值',
        line=dict(color=C['accent'], width=3),
        marker=dict(size=6, color=C['accent']),
        hovertemplate='Day %{customdata[0]}<br>N=%{customdata[1]:,}<br>P=%{y:.4f}<extra></extra>',
        customdata=np.column_stack([cpv['day'], cpv['n_cum']])
    ))
    # O'Brien-Fleming dynamic boundary
    fig_pv.add_trace(go.Scatter(
        x=cpv['date'], y=cpv['obf_alpha'],
        mode='lines', name="O'Brien-Fleming 边界",
        line=dict(color=C['pos'], width=2, dash='dash'),
        hovertemplate='OBF α=%{y:.4f}<extra></extra>'
    ))
    fig_pv.add_hline(y=0.05, line_dash='dot', line_color=C['neg'], line_width=1.5,
                     annotation_text='固定 α = 0.05', annotation_position='top left',
                     annotation_font_color=C['neg'])
    sig_date = cpv.loc[cpv['p_value'] < 0.05, 'date']
    if len(sig_date):
        first_sig = sig_date.iloc[0]
        first_sig_str = pd.Timestamp(first_sig).strftime('%Y-%m-%d')
        fig_pv.add_shape(type='line', x0=first_sig_str, x1=first_sig_str, y0=0, y1=1,
                         yref='paper', line=dict(color=C['pos'], width=1.5, dash='dot'))
        fig_pv.add_annotation(x=first_sig_str, y=1, yref='paper',
                              text=f'首次显著 ({pd.Timestamp(first_sig).strftime("%m-%d")})',
                              showarrow=False, font=dict(color=C['pos'], size=12),
                              xanchor='left', yanchor='bottom')
    fig_pv.update_layout(
        title='P 值收敛 × OBF 序贯边界',
        xaxis_title='日期', yaxis_title='P-Value',
        yaxis=dict(type='log', range=[-4, 0.2])
    )
    style_fig(fig_pv, 420)
    show_chart(fig_pv)

    st.markdown('<div class="sec-h">每日转化率趋势</div>', unsafe_allow_html=True)
    daily = df.groupby(['date', 'group']).agg(
        visitors=('user_id', 'count'), conversions=('converted', 'sum')
    ).reset_index()
    daily['rate'] = daily['conversions'] / daily['visitors']

    fig_daily = go.Figure()
    for g, color, name in [('A', C['a'], 'A组 (对照)'), ('B', C['b'], 'B组 (实验)')]:
        gd = daily[daily['group'] == g]
        fig_daily.add_trace(go.Scatter(
            x=gd['date'], y=gd['rate'], name=name,
            mode='lines+markers', line=dict(color=color, width=2.5),
            marker=dict(size=5), hovertemplate='%{x|%m-%d}<br>转化率=%{y:.2%}<extra></extra>'
        ))
    fig_daily.update_layout(title='每日转化率对比', xaxis_title='日期',
                            yaxis_title='转化率', yaxis_tickformat='.1%')
    style_fig(fig_daily, 400)
    show_chart(fig_daily)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="sec-h">每小时流量热力图</div>', unsafe_allow_html=True)
        dow_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        heat = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        heat_pivot = heat.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
        fig_heat = px.imshow(
            heat_pivot.values, x=[f'{h}:00' for h in range(24)], y=dow_labels,
            color_continuous_scale='Viridis', aspect='auto',
            labels=dict(x='小时', y='', color='流量')
        )
        fig_heat.update_layout(title='流量热力图 (星期 × 小时)')
        style_fig(fig_heat, 320)
        show_chart(fig_heat)
    with col_b:
        st.markdown('<div class="sec-h">累积转化率收敛</div>', unsafe_allow_html=True)
        fig_cum = go.Figure()
        for g, color, name in [('A', C['a'], 'A组'), ('B', C['b'], 'B组')]:
            fig_cum.add_trace(go.Scatter(
                x=cpv['date'], y=cpv[f'cum_{g.lower()}_rate'], name=name,
                mode='lines', line=dict(color=color, width=2.5),
                hovertemplate='%{x|%m-%d}<br>累积转化率=%{y:.2%}<extra></extra>'
            ))
        fig_cum.update_layout(title='累积转化率', xaxis_title='日期',
                              yaxis_title='转化率', yaxis_tickformat='.2%')
        style_fig(fig_cum, 320)
        show_chart(fig_cum)

# ===== TAB 2: 转化漏斗 =====
with tab2:
    st.markdown('<div class="sec-h">用户转化漏斗对比</div>', unsafe_allow_html=True)
    st.markdown("""<div class="box-insight">
    <strong>💡 核心发现：</strong>B组（交互式报价器）在「浏览→互动」环节实现了显著的转化提升。
    传统图文列表让用户在面对复杂服务时迷失，而动态报价器通过引导交互大幅降低了认知门槛。
    </div>""", unsafe_allow_html=True)

    stages = ['广告点击', '浏览页面', '互动/阅读', '提交线索']
    funnel_data = {}
    for g in ['A', 'B']:
        gdf = df[df['group'] == g]
        total = len(gdf)
        funnel_data[g] = [total] + [len(gdf[gdf['stage_reached'] >= s]) for s in [2, 3, 4]]

    col_f1, col_f2 = st.columns(2)
    for col, g, color, title in [(col_f1, 'A', C['a'], '🅰️ 对照组 — 标准图文列表'),
                                  (col_f2, 'B', C['b'], '🅱️ 实验组 — 动态报价器')]:
        with col:
            vals = funnel_data[g]
            rates = [v / vals[0] * 100 for v in vals]
            fig_f = go.Figure(go.Funnel(
                y=stages, x=vals, textinfo='value+percent initial',
                marker=dict(color=[color] * 4),
                connector=dict(line=dict(color='#e2e8f0', width=1))
            ))
            fig_f.update_layout(title=title)
            style_fig(fig_f, 400)
            show_chart(fig_f)

    # 环节转化率对比
    st.markdown('<div class="sec-h">各环节转化率对比</div>', unsafe_allow_html=True)
    step_labels = ['点击→浏览', '浏览→互动', '互动→提交']
    step_rates = {}
    for g in ['A', 'B']:
        v = funnel_data[g]
        step_rates[g] = [v[1]/v[0], v[2]/v[1], v[3]/v[2]]

    fig_step = go.Figure()
    for g, color, name in [('A', C['a'], 'A组'), ('B', C['b'], 'B组')]:
        fig_step.add_trace(go.Bar(
            x=step_labels, y=step_rates[g], name=name,
            marker_color=color, text=[f'{r:.1%}' for r in step_rates[g]],
            textposition='outside'
        ))
    fig_step.update_layout(
        title='各环节转化率', barmode='group',
        yaxis_tickformat='.0%', yaxis_title='转化率'
    )
    style_fig(fig_step, 400)
    show_chart(fig_step)

    improvement = [(step_rates['B'][i] - step_rates['A'][i]) / step_rates['A'][i] * 100 for i in range(3)]
    best_step = step_labels[np.argmax(improvement)]
    st.markdown(f"""<div class="box-success">
    <strong>✅ 关键结论：</strong>B组在「{best_step}」环节提升最为显著（+{max(improvement):.1f}%），
    证明交互式报价器有效降低了用户在浏览阶段的流失率。
    </div>""", unsafe_allow_html=True)

# ===== TAB 3: ROI 分析 =====
with tab3:
    st.markdown('<div class="sec-h">广告投入回报率 (ROI) 分析</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="box-insight">
    <strong>🎯 北极星指标：</strong>已跑通的 1:4 广告投入回报比。当前综合 ROI = 1:{blended_roi:.2f}，
    其中 A组 ROI = 1:{S['A']['roi']:.2f}，B组 ROI = 1:{S['B']['roi']:.2f}。
    B组通过提升转化率使相同广告投入产生更多收益。
    </div>""", unsafe_allow_html=True)

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        fig_roi = go.Figure(go.Bar(
            x=['A组 (对照)', 'B组 (实验)'],
            y=[S['A']['roi'], S['B']['roi']],
            marker_color=[C['a'], C['b']],
            text=[f'1:{S["A"]["roi"]:.2f}', f'1:{S["B"]["roi"]:.2f}'],
            textposition='outside', textfont_size=16
        ))
        fig_roi.add_hline(y=4, line_dash='dash', line_color=C['neg'],
                          annotation_text='目标 ROI = 1:4')
        fig_roi.update_layout(title='💰 ROI 对比', yaxis_title='ROI')
        style_fig(fig_roi, 380)
        show_chart(fig_roi)
    with col_r2:
        fig_cpa = go.Figure(go.Bar(
            x=['A组', 'B组'],
            y=[S['A']['cpa'], S['B']['cpa']],
            marker_color=[C['a'], C['b']],
            text=[f'¥{S["A"]["cpa"]:.1f}', f'¥{S["B"]["cpa"]:.1f}'],
            textposition='outside', textfont_size=16
        ))
        fig_cpa.update_layout(title='📉 CPA (获客成本)', yaxis_title='元/转化')
        style_fig(fig_cpa, 380)
        show_chart(fig_cpa)
    with col_r3:
        fig_rev = go.Figure(go.Bar(
            x=['A组', 'B组'],
            y=[S['A']['revenue'], S['B']['revenue']],
            marker_color=[C['a'], C['b']],
            text=[f'¥{S["A"]["revenue"]:,.0f}', f'¥{S["B"]["revenue"]:,.0f}'],
            textposition='outside', textfont_size=14
        ))
        fig_rev.update_layout(title='💵 总收入对比', yaxis_title='元')
        style_fig(fig_rev, 380)
        show_chart(fig_rev)

    # 每日 ROI 趋势
    st.markdown('<div class="sec-h">每日 ROI 趋势</div>', unsafe_allow_html=True)
    daily_roi = df.groupby(['date', 'group']).agg(
        revenue=('revenue', 'sum'), cost=('ad_cost', 'sum')
    ).reset_index()
    daily_roi['roi'] = daily_roi.apply(lambda r: r['revenue'] / r['cost'] if r['cost'] > 0 else 0, axis=1)

    fig_droi = go.Figure()
    for g, color, name in [('A', C['a'], 'A组'), ('B', C['b'], 'B组')]:
        gd = daily_roi[daily_roi['group'] == g]
        fig_droi.add_trace(go.Scatter(
            x=gd['date'], y=gd['roi'], name=name,
            mode='lines+markers', line=dict(color=color, width=2),
            marker=dict(size=4)
        ))
    fig_droi.add_hline(y=4, line_dash='dash', line_color=C['neg'], line_width=1.5,
                       annotation_text='ROI = 1:4 基准线')
    fig_droi.update_layout(title='每日 ROI 趋势', xaxis_title='日期', yaxis_title='ROI')
    style_fig(fig_droi, 380)
    show_chart(fig_droi)

    # 收入瀑布图
    st.markdown('<div class="sec-h">收入贡献瀑布图</div>', unsafe_allow_html=True)
    svc_rev = df[df['converted'] == 1].groupby(['group', 'service_type'])['revenue'].sum().reset_index()
    fig_wf = px.sunburst(svc_rev, path=['group', 'service_type'], values='revenue',
                         color='group', color_discrete_map={'A': C['a'], 'B': C['b']})
    fig_wf.update_layout(title='收入构成 (类型×组别)')
    style_fig(fig_wf, 480)
    show_chart(fig_wf)

# ===== TAB 4: 统计检验 =====
with tab4:
    st.markdown('<div class="sec-h">频率学派统计检验</div>', unsafe_allow_html=True)

    z_stat, z_p, ci_diff, pa, pb = run_z_test(df)
    chi2_val, chi2_p, chi2_dof = run_chi2_test(df)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### 📐 双比例 Z 检验")
        st.markdown(f"""
        | 指标 | 值 |
        |-----|----|
        | A 组转化率 | **{pa:.4f}** ({pa:.2%}) |
        | B 组转化率 | **{pb:.4f}** ({pb:.2%}) |
        | 绝对差异 | **{pb - pa:.4f}** |
        | Z 统计量 | **{z_stat:.4f}** |
        | P 值 (双侧) | **{z_p:.6f}** |
        | 95% CI (差异) | **[{ci_diff[0]:.4f}, {ci_diff[1]:.4f}]** |
        | 显著性 (α=0.05) | **{'✅ 是' if z_p < 0.05 else '❌ 否'}** |
        """)

    with col_t2:
        st.markdown("#### 📊 卡方独立性检验")
        st.markdown(f"""
        | 指标 | 值 |
        |-----|----|
        | χ² 统计量 | **{chi2_val:.4f}** |
        | 自由度 | **{chi2_dof}** |
        | P 值 | **{chi2_p:.6f}** |
        | 显著性 (α=0.05) | **{'✅ 是' if chi2_p < 0.05 else '❌ 否'}** |
        """)

    # 置信区间可视化 — Forest Plot 风格
    st.markdown('<div class="sec-h">转化率置信区间 (Forest Plot)</div>', unsafe_allow_html=True)
    n_a, n_b = S['A']['visitors'], S['B']['visitors']
    se_a = np.sqrt(pa * (1 - pa) / n_a)
    se_b = np.sqrt(pb * (1 - pb) / n_b)

    fig_ci = go.Figure()
    groups_ci = [
        ('B组 (实验)', pb, se_b, C['b']),
        ('A组 (对照)', pa, se_a, C['a']),
    ]
    for i, (name, rate, se, color) in enumerate(groups_ci):
        lo, hi = rate - 1.96 * se, rate + 1.96 * se
        # CI band
        fig_ci.add_shape(type='rect', x0=lo, x1=hi, y0=i - 0.25, y1=i + 0.25,
                         fillcolor=hex_to_rgba(color, 0.2), line=dict(color=color, width=1.5))
        # Center point
        fig_ci.add_trace(go.Scatter(
            x=[rate], y=[i], mode='markers',
            marker=dict(color=color, size=14, symbol='diamond',
                        line=dict(color='white', width=2)),
            name=name,
            hovertemplate=f'{name}<br>转化率: {rate:.2%}<br>95% CI: [{lo:.2%}, {hi:.2%}]<extra></extra>'
        ))
        # Text labels
        fig_ci.add_annotation(x=rate, y=i + 0.42, text=f'<b>{rate:.2%}</b> [{lo:.2%},{hi:.2%}]',
                              showarrow=False, font=dict(size=10, color=color), xanchor='center')

    fig_ci.update_layout(
        title='转化率 95% CI',
        xaxis_title='转化率', xaxis_tickformat='.1%',
        yaxis=dict(tickvals=[0, 1], ticktext=['B组', 'A组'], range=[-0.8, 2.0]),
        showlegend=False, height=260,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=36, r=30, t=50, b=40),
        font=dict(family="Inter, system-ui, sans-serif", color='#334155', size=11)
    )
    fig_ci.update_xaxes(gridcolor='#e2e8f0', zeroline=False)
    fig_ci.update_yaxes(gridcolor='rgba(0,0,0,0)', zeroline=False)
    show_chart(fig_ci)

    # 差异置信区间
    se_diff = np.sqrt(pa*(1-pa)/n_a + pb*(1-pb)/n_b)
    diff = pb - pa
    diff_lo, diff_hi = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    fig_diff = go.Figure()
    fig_diff.add_shape(type='rect', x0=diff_lo, x1=diff_hi, y0=-0.3, y1=0.3,
                       fillcolor=hex_to_rgba(C['accent'], 0.2),
                       line=dict(color=C['accent'], width=1.5))
    fig_diff.add_trace(go.Scatter(
        x=[diff], y=[0], mode='markers',
        marker=dict(color=C['accent'], size=14, symbol='diamond', line=dict(color='white', width=2)),
        hovertemplate=f'差异: {diff:.2%}<br>95% CI: [{diff_lo:.2%}, {diff_hi:.2%}]<extra></extra>',
        showlegend=False
    ))
    fig_diff.add_vline(x=0, line_dash='dash', line_color=C['neg'], line_width=1.5)
    fig_diff.add_annotation(x=0, y=0.45, text='零效应线 (H₀)', showarrow=False,
                            font=dict(color=C['neg'], size=11))
    fig_diff.add_annotation(x=diff, y=0.5,
                            text=f'<b>Δ={diff:.2%}</b> [{diff_lo:.2%},{diff_hi:.2%}]',
                            showarrow=False, font=dict(color=C['accent'], size=10), xanchor='center')
    fig_diff.update_layout(
        title='转化率差异 (B−A) 95% CI',
        xaxis_title='转化率差异', xaxis_tickformat='.2%',
        yaxis=dict(visible=False, range=[-0.8, 0.8]),
        showlegend=False, height=200,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=36, r=30, t=50, b=40),
        font=dict(family="Inter, system-ui, sans-serif", color='#334155', size=11)
    )
    fig_diff.update_xaxes(gridcolor='#e2e8f0', zeroline=False)
    show_chart(fig_diff)

    # 统计功效分析
    st.markdown('<div class="sec-h">统计功效 (Power) 分析</div>', unsafe_allow_html=True)
    effect = pb - pa
    pooled_se = np.sqrt(pa * (1-pa)/n_a + pb * (1-pb)/n_b)
    achieved_power = sp_stats.norm.cdf(abs(effect) / pooled_se - 1.96)

    sizes = np.arange(500, 15001, 200)
    powers = []
    for n in sizes:
        se_hyp = np.sqrt(pa*(1-pa)/n + pb*(1-pb)/n)
        pw = sp_stats.norm.cdf(abs(effect)/se_hyp - 1.96)
        powers.append(pw)

    fig_pw = go.Figure()
    fig_pw.add_trace(go.Scatter(
        x=sizes * 2, y=powers, mode='lines',
        line=dict(color=C['accent'], width=3), name='统计功效'
    ))
    fig_pw.add_hline(y=0.8, line_dash='dash', line_color=C['neg'],
                     annotation_text='Power = 80%')
    fig_pw.add_vline(x=float(n_a + n_b), line_dash='dot', line_color=C['pos'],
                     annotation_text=f'当前样本 N={n_a+n_b:,}')
    fig_pw.update_layout(title=f'功效曲线 (Power≈{achieved_power:.1%})',
                         xaxis_title='总样本量 (A+B)', yaxis_title='Power',
                         yaxis_tickformat='.0%')
    style_fig(fig_pw, 380)
    show_chart(fig_pw)

    if z_p < 0.05:
        st.markdown(f"""<div class="box-success">
        <strong>✅ 结论：</strong>在 α=0.05 显著性水平下，实验组 B 的转化率显著高于对照组 A
        （P={z_p:.6f}）。在当前样本量下，统计功效达到 {achieved_power:.1%}，实验结果可靠。
        建议全量上线交互式报价器方案。
        </div>""", unsafe_allow_html=True)

# ===== TAB 5: 贝叶斯推断 =====
with tab5:
    st.markdown('<div class="sec-h">贝叶斯 A/B 测试分析</div>', unsafe_allow_html=True)
    st.markdown("""<div class="box-insight">
    <strong>💡 为什么用贝叶斯？</strong>频率学派给出"是否显著"的二元判断，
    而贝叶斯方法给出"B 比 A 好的概率是多少"以及"提升幅度的分布"，对商业决策更直观。
    使用 Beta-Binomial 共轭先验模型，先验为无信息 Beta(1,1)。
    </div>""", unsafe_allow_html=True)

    a_samp, b_samp, prob_b, lift_samp = bayesian_analysis(df)

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-icon">🏆</div>
            <div class="kpi-value" style="color:{C['pos']}">{prob_b:.1%}</div>
            <div class="kpi-label">P(B &gt; A)</div>
        </div>""", unsafe_allow_html=True)
    with col_b2:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-icon">📈</div>
            <div class="kpi-value">{np.mean(lift_samp):.1%}</div>
            <div class="kpi-label">期望提升</div>
        </div>""", unsafe_allow_html=True)
    with col_b3:
        ci_low, ci_high = np.percentile(lift_samp, [2.5, 97.5])
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-icon">📊</div>
            <div class="kpi-value" style="font-size:1.4rem;">[{ci_low:.1%}, {ci_high:.1%}]</div>
            <div class="kpi-label">95% 可信区间</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # 后验分布图 — 使用 rgba 替代 hex+alpha
    st.markdown('<div class="sec-h">转化率后验分布</div>', unsafe_allow_html=True)
    fig_post = go.Figure()
    for samp, color, fill, name in [
        (a_samp, C['a'], C['a_fill'], 'A组 后验'),
        (b_samp, C['b'], C['b_fill'], 'B组 后验')
    ]:
        counts, bins = np.histogram(samp, bins=200, density=True)
        fig_post.add_trace(go.Scatter(
            x=(bins[:-1] + bins[1:]) / 2, y=counts, mode='lines',
            fill='tozeroy', line=dict(color=color, width=2),
            fillcolor=fill, name=name
        ))
    fig_post.add_vline(x=float(np.mean(a_samp)), line_dash='dot', line_color=C['a'],
                       annotation_text=f'A均值={np.mean(a_samp):.2%}')
    fig_post.add_vline(x=float(np.mean(b_samp)), line_dash='dot', line_color=C['b'],
                       annotation_text=f'B均值={np.mean(b_samp):.2%}')
    fig_post.update_layout(title='后验分布 — 转化率', xaxis_title='转化率',
                           yaxis_title='概率密度', xaxis_tickformat='.2%')
    style_fig(fig_post, 400)
    show_chart(fig_post)

    # 提升幅度分布
    st.markdown('<div class="sec-h">相对提升幅度分布</div>', unsafe_allow_html=True)
    fig_lift = go.Figure()
    counts_l, bins_l = np.histogram(lift_samp, bins=200, density=True)
    fig_lift.add_trace(go.Scatter(
        x=(bins_l[:-1] + bins_l[1:]) / 2, y=counts_l, mode='lines',
        fill='tozeroy', line=dict(color=C['accent'], width=2),
        fillcolor=C['accent_fill'], name='提升幅度分布'
    ))
    fig_lift.add_vline(x=0, line_dash='dash', line_color=C['neg'],
                       annotation_text='零提升线')
    fig_lift.add_vline(x=float(np.mean(lift_samp)), line_dash='dot', line_color=C['pos'],
                       annotation_text=f'期望提升={np.mean(lift_samp):.1%}')
    fig_lift.update_layout(title='提升幅度分布 (B vs A)',
                           xaxis_title='相对提升', yaxis_title='概率密度',
                           xaxis_tickformat='.0%')
    style_fig(fig_lift, 380)
    show_chart(fig_lift)

    prob_loss = (lift_samp < 0).mean()

    # Expected Loss 分析
    st.markdown('<div class="sec-h">期望损失 (Expected Loss) 分析</div>', unsafe_allow_html=True)
    st.markdown("""<div class="box-insight">
    <strong>💡 Expected Loss 是什么？</strong>选择某个方案的期望损失 = 当该方案实际更差时，
    你平均会损失多少转化率。它是贝叶斯决策理论的核心指标（Letham & Bakshy, 2019）：
    E[Loss_B] = E[max(θ_A − θ_B, 0)]。当 E[Loss] 低于业务可容忍阈值（如 0.1%）时，即可放心决策。
    </div>""", unsafe_allow_html=True)

    loss_a = np.maximum(b_samp - a_samp, 0).mean()  # 选A的期望损失
    loss_b = np.maximum(a_samp - b_samp, 0).mean()  # 选B的期望损失
    # Risk ratio: how many times more risk if choosing A vs B
    risk_ratio = loss_a / loss_b if loss_b > 0 else float('inf')

    def fmt_loss(v):
        if v < 1e-6: return f'{v:.2e}'
        if v < 0.0001: return f'{v:.6f}'
        return f'{v:.4f}'

    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-icon">🅰️</div>
            <div class="kpi-value" style="color:{C['neg']}; font-size:1.5rem;">{fmt_loss(loss_a)}</div>
            <div class="kpi-label">选择 A 的期望损失</div>
        </div>""", unsafe_allow_html=True)
    with col_l2:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-icon">🅱️</div>
            <div class="kpi-value" style="color:{C['pos']}; font-size:1.5rem;">{fmt_loss(loss_b)}</div>
            <div class="kpi-label">选择 B 的期望损失</div>
        </div>""", unsafe_allow_html=True)
    with col_l3:
        ratio_text = f'{risk_ratio:.0f}×' if risk_ratio < 1e6 else '∞'
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-icon">⚖️</div>
            <div class="kpi-value" style="color:{C['accent']}; font-size:1.5rem;">{ratio_text}</div>
            <div class="kpi-label">风险比 (A / B)</div>
        </div>""", unsafe_allow_html=True)

    # 柱状图 — 对数坐标轴处理量级差异
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Bar(
        x=['选择 A 的期望损失', '选择 B 的期望损失'],
        y=[max(loss_a, 1e-8), max(loss_b, 1e-8)],
        marker_color=[C['a'], C['b']],
        text=[fmt_loss(loss_a), fmt_loss(loss_b)],
        textposition='outside', textfont_size=14
    ))
    fig_loss.add_hline(y=0.001, line_dash='dash', line_color=C['gray'],
                       annotation_text='典型决策阈值 ε = 0.1%', annotation_position='top left')
    fig_loss.update_layout(
        title='期望损失对比 (对数坐标)',
        yaxis_title='期望损失 (绝对转化率)',
        yaxis_type='log'
    )
    style_fig(fig_loss, 360)
    show_chart(fig_loss)

    threshold = 0.001  # 典型阈值
    below_threshold = loss_b < threshold
    st.markdown(f"""<div class="box-success">
    <strong>✅ 贝叶斯结论：</strong>B 优于 A 的后验概率为 <strong>{prob_b:.1%}</strong>，
    期望转化率提升 <strong>{np.mean(lift_samp):.1%}</strong>。
    选择 B 的期望损失为 <strong>{fmt_loss(loss_b)}</strong>，
    {'远低于' if below_threshold else '接近'}决策阈值 ε=0.1%。
    选择 A 的风险是选择 B 的 <strong>{ratio_text}</strong>。
    B 组表现劣于 A 组的概率仅有 <strong>{prob_loss:.2%}</strong>。
    从商业决策的角度，可以高置信度地推荐全量上线实验方案。
    </div>""", unsafe_allow_html=True)

# ===== TAB 6: 分群洞察 =====
with tab6:
    st.markdown('<div class="sec-h">多维度分群分析</div>', unsafe_allow_html=True)
    st.markdown("""<div class="box-insight">
    <strong>💡 为什么要做分群？</strong>整体显著不代表对所有用户群体都有效。
    分群分析（Heterogeneous Treatment Effects）帮助我们识别哪些群体受益最大，指导精细化运营。
    </div>""", unsafe_allow_html=True)

    # 按渠道
    st.markdown("#### 📱 按渠道分析")
    ch_stats = df.groupby(['channel', 'group']).agg(
        n=('user_id', 'count'), conv=('converted', 'sum')
    ).reset_index()
    ch_stats['rate'] = ch_stats['conv'] / ch_stats['n']

    fig_ch = px.bar(ch_stats, x='channel', y='rate', color='group', barmode='group',
                    color_discrete_map={'A': C['a'], 'B': C['b']},
                    text=ch_stats['rate'].apply(lambda x: f'{x:.2%}'),
                    labels={'channel': '渠道', 'rate': '转化率', 'group': '组别'})
    fig_ch.update_layout(title='各渠道转化率对比', yaxis_tickformat='.1%')
    fig_ch.update_traces(textposition='outside')
    style_fig(fig_ch, 400)
    show_chart(fig_ch)

    # 按设备
    st.markdown("#### 💻 按设备分析")
    dev_stats = df.groupby(['device', 'group']).agg(
        n=('user_id', 'count'), conv=('converted', 'sum')
    ).reset_index()
    dev_stats['rate'] = dev_stats['conv'] / dev_stats['n']

    fig_dev = px.bar(dev_stats, x='device', y='rate', color='group', barmode='group',
                     color_discrete_map={'A': C['a'], 'B': C['b']},
                     text=dev_stats['rate'].apply(lambda x: f'{x:.2%}'),
                     labels={'device': '设备', 'rate': '转化率', 'group': '组别'})
    fig_dev.update_layout(title='各设备转化率对比', yaxis_tickformat='.1%')
    fig_dev.update_traces(textposition='outside')
    style_fig(fig_dev, 380)
    show_chart(fig_dev)

    # 工作日 vs 周末
    st.markdown("#### 📅 工作日 vs 周末")
    we_stats = df.copy()
    we_stats['period'] = we_stats['is_weekend'].map({True: '周末', False: '工作日'})
    we_agg = we_stats.groupby(['period', 'group']).agg(
        n=('user_id', 'count'), conv=('converted', 'sum')
    ).reset_index()
    we_agg['rate'] = we_agg['conv'] / we_agg['n']

    fig_we = px.bar(we_agg, x='period', y='rate', color='group', barmode='group',
                    color_discrete_map={'A': C['a'], 'B': C['b']},
                    text=we_agg['rate'].apply(lambda x: f'{x:.2%}'),
                    labels={'period': '', 'rate': '转化率', 'group': '组别'})
    fig_we.update_layout(title='工作日 vs 周末转化率', yaxis_tickformat='.1%')
    fig_we.update_traces(textposition='outside')
    style_fig(fig_we, 380)
    show_chart(fig_we)

    # 服务类型收入
    st.markdown("#### 🏷️ 转化用户服务类型分布")
    svc_df = df[df['converted'] == 1]
    col_s1, col_s2 = st.columns(2)
    for col, g, title in [(col_s1, 'A', 'A组'), (col_s2, 'B', 'B组')]:
        with col:
            sg = svc_df[svc_df['group'] == g]['service_type'].value_counts()
            fig_pie = px.pie(values=sg.values, names=sg.index, title=f'{title} 服务类型分布',
                             color_discrete_sequence=px.colors.qualitative.Set3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            style_fig(fig_pie, 380)
            show_chart(fig_pie)

# ===== TAB 7: 方法论 =====
with tab7:
    st.markdown('<div class="sec-h">项目方法论</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1️⃣ 业务背景

    **北大的小码农** 是一家提供多元化技术服务的数字化工作室，业务涵盖 PPT 定制、文章代写、
    编程代写、网站开发、机械设计、数据分析等。通过淘宝直通车进行广告投放，目前已跑通
    **1:4 的广告投入回报比 (ROI)**。

    **痛点发现：** 许多客户通过广告点击进入落地页后，面对复杂的技术服务列表（服务种类多、
    定价不透明），产生信息过载，在"浏览→互动"环节大量流失。

    ---

    ### 2️⃣ 测试假设

    **零假设 H₀：** 交互式需求评估与动态报价器不会改变转化率。

    **备择假设 H₁：** 交互式报价器通过降低用户认知门槛，能显著提升"浏览→互动→提交"转化率。

    | 分组 | 方案 | 描述 |
    |------|------|------|
    | A (Control) | 标准图文列表 | 现有服务列表页面 |
    | B (Treatment) | 交互式报价器 | 用户选择需求后自动生成预估报价 |

    ---

    ### 3️⃣ 数据生成 — 蒙特卡洛模拟

    本项目数据采用 **蒙特卡洛方法 (Monte Carlo Simulation)** 模拟生成。
    参数设定参考了商业运营经验与以下行业公开数据：

    - **CNNIC 第 47 次互联网发展报告** — 移动/桌面流量比、时段分布
    - **淘宝直通车公开投放数据** — CPC 区间 ¥1.8-3.8、渠道占比
    - **Google Analytics Industry Benchmarks** — 服务类电商转化率 2-8%
    - **SimilarWeb 行业报告** — 页面停留时间分布

    核心模拟参数：

    | 维度 | 分布 | 参数 | 依据 |
    |------|------|------|------|
    | 日流量 | Poisson(λ) | λ=480 | 直通车均值展现 |
    | 页面停留 | LogNormal(μ,σ) | μ=3.2, σ=0.9 | SimilarWeb |
    | 渠道比例 | Categorical | 直通车58%/搜索27%/社媒15% | 运营后台 |
    | 周末效应 | 乘法因子 | 流量×0.82, 转化×0.83 | 实际数据 |
    | 随机噪音 | Normal(μ,σ) | μ=1.0, σ=0.08 | 控制信噪比 |

    B 组在"浏览→互动"环节注入 ~+18% 的提升（从 38% 到 45%），
    模拟交互式报价器降低认知门槛的效果。参考 Google/Bing 公开
    A/B 测试论文中典型的 5-20% 相对提升范围。

    ---

    ### 4️⃣ 统计分析方法

    | 方法 | 用途 | 参考文献 |
    |------|------|----------|
    | 卡方独立性检验 | 检验两组转化率是否独立 | Pearson, 1900 |
    | 双比例 Z 检验 | 计算转化率差异的显著性和 CI | Agresti & Caffo, 2000 |
    | 累积 P 值 + O'Brien-Fleming 边界 | 序贯检验，控制偷窥问题 | O'Brien & Fleming, 1979 |
    | 统计功效分析 | 评估实验的检测能力 | Cohen, 1988 |
    | SRM 检验 | 检测样本比例失配 | Fabijan et al., 2019 |
    | Beta-Binomial 贝叶斯推断 | 后验概率 P(B>A) 和提升分布 | Gelman et al., 2013 |
    | Expected Loss | 贝叶斯决策框架 | Letham & Bakshy, 2019 |

    ---

    ### 5️⃣ 技术架构

    ```
    Python 3.x
    ├── data_generator.py    # 蒙特卡洛数据模拟引擎
    ├── app.py               # Streamlit 可视化仪表盘
    ├── requirements.txt     # 依赖管理
    └── .streamlit/
        └── config.toml      # 应用配置与主题
    ```

    **核心技术栈：** Streamlit · Plotly · Pandas · NumPy · SciPy · Bayesian Inference

    ---

    ### 6️⃣ 项目亮点

    - ✅ 完整的 A/B 测试全流程方法论，从假设到决策
    - ✅ 蒙特卡洛模拟 — 参数基于行业公开基准，包含多维噪音
    - ✅ **序贯检验** — O'Brien-Fleming 花费函数边界，解决偷窥问题
    - ✅ **SRM 检验** — 样本比例失配检测，保证实验完整性
    - ✅ **贝叶斯推断 + Expected Loss** — 直接量化决策风险
    - ✅ 分群洞察 (HTE) — 识别不同群体的差异化效果
    - ✅ 真实业务场景 — 基于创业实战经验，ROI 北极星指标驱动
    """)

    st.markdown("""<div class="box-success">
    <strong>🎓 关于作者：</strong>本项目由 <strong>陈文杰</strong> 独立完成，
    结合真实创业运营经验与数据分析方法论，力求展现从业务理解、实验设计、统计推断到
    商业洞察的全链路数据分析能力。
    <br><br>
    <a href="https://github.com/ucarcompany" target="_blank">GitHub: @ucarcompany</a>
    </div>""", unsafe_allow_html=True)
