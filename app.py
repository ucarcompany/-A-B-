"""
数字化服务工作室转化链路与动态报价 A/B 测试分析平台
===================================================
北大的小码农 | Monte Carlo Simulation + Statistical Inference + Bayesian Analysis
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from data_generator import generate_ab_test_data, get_summary_stats

# ==================================================
# 页面配置
# ==================================================
st.set_page_config(
    page_title="A/B Test | 北大的小码农",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
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
    'accent': '#8b5cf6'
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
def style_fig(fig, height=440):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=height, margin=dict(l=50, r=30, t=50, b=40),
        font=dict(family="Inter, system-ui, sans-serif", color='#334155', size=13),
        legend=dict(bgcolor='rgba(255,255,255,.8)', bordercolor='rgba(0,0,0,.08)',
                    borderwidth=1, font_size=12),
        hovermode='x unified'
    )
    fig.update_xaxes(gridcolor='#e2e8f0', zeroline=False)
    fig.update_yaxes(gridcolor='#e2e8f0', zeroline=False)
    return fig

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
        records.append({'date': d, 'p_value': p, 'day': i + 1,
                        'cum_a_rate': a['converted'].mean(),
                        'cum_b_rate': b['converted'].mean()})
    return pd.DataFrame(records)

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

    数据基于真实商业经验，采用 **蒙特卡洛方法** 模拟生成，包含季节性波动与随机噪音。
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
        <p style="font-weight:700;">北大的小码农</p>
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
    st.markdown('<div class="sec-h">累积 P 值收敛图 — 识别「偷窥陷阱」</div>', unsafe_allow_html=True)
    st.markdown("""<div class="box-warn">
    <strong>⚠️ Peeking Problem（偷窥陷阱）：</strong>在实验期间反复查看 P 值会严重膨胀假阳性率。
    下图展示 P 值随样本量增长的收敛过程：只有当曲线稳定地落在 α=0.05 以下并不再反复穿越时，
    才可以可靠地拒绝零假设。这正是字节跳动等大厂面试中对 A/B 测试理解的核心考点。
    </div>""", unsafe_allow_html=True)

    cpv = cumulative_pvalues(df)
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scatter(
        x=cpv['date'], y=cpv['p_value'],
        mode='lines+markers', name='累积 P 值',
        line=dict(color=C['accent'], width=3),
        marker=dict(size=6, color=C['accent']),
        hovertemplate='Day %{customdata}<br>P=%{y:.4f}<extra></extra>',
        customdata=cpv['day']
    ))
    fig_pv.add_hline(y=0.05, line_dash='dash', line_color=C['neg'], line_width=2,
                     annotation_text='α = 0.05', annotation_position='top left',
                     annotation_font_color=C['neg'])
    sig_date = cpv.loc[cpv['p_value'] < 0.05, 'date']
    if len(sig_date):
        first_sig = sig_date.iloc[0]
        fig_pv.add_vline(x=first_sig, line_dash='dot', line_color=C['pos'], line_width=1.5,
                         annotation_text=f'首次显著 ({pd.Timestamp(first_sig).strftime("%m-%d")})',
                         annotation_position='top right',
                         annotation_font_color=C['pos'])
    fig_pv.update_layout(
        title='累积 P 值收敛趋势',
        xaxis_title='日期', yaxis_title='P-Value',
        yaxis=dict(type='log', range=[-4, 0.2])
    )
    style_fig(fig_pv, 420)
    st.plotly_chart(fig_pv, use_container_width=True)

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
    st.plotly_chart(fig_daily, use_container_width=True)

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
        st.plotly_chart(fig_heat, use_container_width=True)
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
        st.plotly_chart(fig_cum, use_container_width=True)

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
            st.plotly_chart(fig_f, use_container_width=True)

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
    st.plotly_chart(fig_step, use_container_width=True)

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
        st.plotly_chart(fig_roi, use_container_width=True)
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
        st.plotly_chart(fig_cpa, use_container_width=True)
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
        st.plotly_chart(fig_rev, use_container_width=True)

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
    st.plotly_chart(fig_droi, use_container_width=True)

    # 收入瀑布图
    st.markdown('<div class="sec-h">收入贡献瀑布图</div>', unsafe_allow_html=True)
    svc_rev = df[df['converted'] == 1].groupby(['group', 'service_type'])['revenue'].sum().reset_index()
    fig_wf = px.sunburst(svc_rev, path=['group', 'service_type'], values='revenue',
                         color='group', color_discrete_map={'A': C['a'], 'B': C['b']})
    fig_wf.update_layout(title='收入构成 (按服务类型 × 组别)')
    style_fig(fig_wf, 480)
    st.plotly_chart(fig_wf, use_container_width=True)

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

    # 置信区间可视化
    st.markdown('<div class="sec-h">转化率置信区间</div>', unsafe_allow_html=True)
    n_a, n_b = S['A']['visitors'], S['B']['visitors']
    se_a = np.sqrt(pa * (1 - pa) / n_a)
    se_b = np.sqrt(pb * (1 - pb) / n_b)

    fig_ci = go.Figure()
    for g, rate, se, color, name in [
        ('A', pa, se_a, C['a'], 'A组 (对照)'),
        ('B', pb, se_b, C['b'], 'B组 (实验)')
    ]:
        fig_ci.add_trace(go.Scatter(
            x=[name], y=[rate], error_y=dict(type='data', array=[1.96*se], color=color, thickness=3, width=12),
            mode='markers', marker=dict(color=color, size=14),
            name=name, hovertemplate=f'{name}<br>转化率={rate:.2%}<br>95%CI=[{rate-1.96*se:.2%}, {rate+1.96*se:.2%}]<extra></extra>'
        ))
    fig_ci.update_layout(title='转化率 95% 置信区间', yaxis_title='转化率', yaxis_tickformat='.2%',
                         showlegend=False)
    style_fig(fig_ci, 360)
    st.plotly_chart(fig_ci, use_container_width=True)

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
    fig_pw.add_vline(x=n_a + n_b, line_dash='dot', line_color=C['pos'],
                     annotation_text=f'当前样本 N={n_a+n_b:,}')
    fig_pw.update_layout(title=f'统计功效曲线 (当前 Power ≈ {achieved_power:.1%})',
                         xaxis_title='总样本量 (A+B)', yaxis_title='Power',
                         yaxis_tickformat='.0%')
    style_fig(fig_pw, 380)
    st.plotly_chart(fig_pw, use_container_width=True)

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

    # 后验分布图
    st.markdown('<div class="sec-h">转化率后验分布</div>', unsafe_allow_html=True)
    fig_post = go.Figure()
    for samp, color, name in [(a_samp, C['a'], 'A组 后验'), (b_samp, C['b'], 'B组 后验')]:
        counts, bins = np.histogram(samp, bins=200, density=True)
        fig_post.add_trace(go.Scatter(
            x=(bins[:-1] + bins[1:]) / 2, y=counts, mode='lines',
            fill='tozeroy', line=dict(color=color, width=2),
            fillcolor=color + '30', name=name
        ))
    fig_post.add_vline(x=np.mean(a_samp), line_dash='dot', line_color=C['a'],
                       annotation_text=f'A均值={np.mean(a_samp):.2%}')
    fig_post.add_vline(x=np.mean(b_samp), line_dash='dot', line_color=C['b'],
                       annotation_text=f'B均值={np.mean(b_samp):.2%}')
    fig_post.update_layout(title='Beta 后验分布 — 转化率', xaxis_title='转化率',
                           yaxis_title='概率密度', xaxis_tickformat='.2%')
    style_fig(fig_post, 400)
    st.plotly_chart(fig_post, use_container_width=True)

    # 提升幅度分布
    st.markdown('<div class="sec-h">相对提升幅度分布</div>', unsafe_allow_html=True)
    fig_lift = go.Figure()
    counts_l, bins_l = np.histogram(lift_samp, bins=200, density=True)
    fig_lift.add_trace(go.Scatter(
        x=(bins_l[:-1] + bins_l[1:]) / 2, y=counts_l, mode='lines',
        fill='tozeroy', line=dict(color=C['accent'], width=2),
        fillcolor=C['accent'] + '30', name='提升幅度分布'
    ))
    fig_lift.add_vline(x=0, line_dash='dash', line_color=C['neg'],
                       annotation_text='零提升线')
    fig_lift.add_vline(x=np.mean(lift_samp), line_dash='dot', line_color=C['pos'],
                       annotation_text=f'期望提升={np.mean(lift_samp):.1%}')
    fig_lift.update_layout(title='B组相对A组的提升幅度后验分布',
                           xaxis_title='相对提升', yaxis_title='概率密度',
                           xaxis_tickformat='.0%')
    style_fig(fig_lift, 380)
    st.plotly_chart(fig_lift, use_container_width=True)

    prob_loss = (lift_samp < 0).mean()
    st.markdown(f"""<div class="box-success">
    <strong>✅ 贝叶斯结论：</strong>B 优于 A 的后验概率为 <strong>{prob_b:.1%}</strong>，
    期望转化率提升 <strong>{np.mean(lift_samp):.1%}</strong>。
    B 组表现劣于 A 组的风险仅有 <strong>{prob_loss:.2%}</strong>。从商业决策的角度，
    可以高置信度地推荐全量上线实验方案。
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
    st.plotly_chart(fig_ch, use_container_width=True)

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
    st.plotly_chart(fig_dev, use_container_width=True)

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
    st.plotly_chart(fig_we, use_container_width=True)

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
            st.plotly_chart(fig_pie, use_container_width=True)

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

    由于本项目为作品集展示，数据采用 **蒙特卡洛方法 (Monte Carlo Simulation)** 模拟生成。
    模拟规则均基于真实商业运营经验：

    - **日流量：** 泊松分布，λ = 480
    - **渠道分布：** 直通车 58% / 自然搜索 27% / 社交媒体 15%
    - **设备分布：** 移动端 68% / 桌面端 32%
    - **周末效应：** 流量 -18%，转化率 -17%
    - **渠道修正：** 自然搜索 +12%，社交媒体 -22%
    - **设备修正：** 移动端 -8%，桌面端 +10%
    - **随机噪音：** 正态分布 N(1.0, 0.08)

    B 组在"浏览→互动"环节注入 ~+37% 的提升（从 38% 到 52%），
    模拟交互式报价器降低认知门槛的效果。最终整体转化率提升约 +1.5 个百分点。

    ---

    ### 4️⃣ 统计分析方法

    | 方法 | 用途 |
    |------|------|
    | 卡方独立性检验 | 检验两组转化率是否独立 |
    | 双比例 Z 检验 | 计算转化率差异的统计显著性和置信区间 |
    | 累积 P 值监控 | 展示偷窥陷阱 (Peeking Problem) |
    | 统计功效分析 | 评估实验的检测能力 |
    | Beta-Binomial 贝叶斯推断 | 计算后验概率 P(B>A) 和提升幅度分布 |

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

    - ✅ 完整的 A/B 测试方法论，从假设设计到统计推断
    - ✅ 蒙特卡洛模拟生成包含季节性波动与多维噪音的真实数据
    - ✅ **累积 P 值收敛图** — 展示对偷窥陷阱 (Peeking Problem) 的深入理解
    - ✅ **贝叶斯推断** — 超越频率学派，给出直观的商业决策概率
    - ✅ 分群洞察 (HTE) — 识别不同用户群体的差异化效果
    - ✅ 真实业务场景 — 基于创业实战经验，ROI 北极星指标驱动
    """)

    st.markdown("""<div class="box-success">
    <strong>🎓 关于作者：</strong>本项目由 <strong>北大的小码农</strong> 独立完成，
    结合真实创业运营经验与数据分析方法论，力求展现从业务理解、实验设计、统计推断到
    商业洞察的全链路数据分析能力。
    <br><br>
    <a href="https://github.com/ucarcompany" target="_blank">GitHub: @ucarcompany</a>
    </div>""", unsafe_allow_html=True)
