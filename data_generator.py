"""
蒙特卡洛模拟数据生成器
=======================
基于真实商业运营经验与行业公开基准数据，采用蒙特卡洛方法模拟生成 A/B 测试数据。
包含季节性波动、渠道差异、设备差异与随机噪音。

参数来源：
- CNNIC 第47次互联网发展报告 — 移动/桌面比 68:32、24h流量分布
- 淘宝直通车公开CPC数据 — ¥1.8~3.8
- Google Analytics Industry Benchmarks — 服务类电商转化率 2-8%
- SimilarWeb — 页面停留时间 LogNormal 分布

业务场景：北大的小码农 - 数字化服务工作室
- 对照组 (A)：标准图文服务列表展示
- 实验组 (B)：交互式需求评估与动态报价器
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_ab_test_data(
    seed: int = 42,
    n_days: int = 30,
    daily_traffic_lambda: int = 480,
    start_date: str = '2025-02-01'
) -> pd.DataFrame:
    """
    使用蒙特卡洛方法生成 A/B 测试模拟日志数据。

    模拟规则：
    - 日均流量服从泊松分布 (λ=480)
    - 不同渠道具有不同的转化率基础概率
    - 周末效应：周末流量下降18%，转化率下降17%
    - 实验组 B 注入约 +1.5% 绝对转化率提升
    - 各环节引入对数正态分布随机噪音
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)

    # === 渠道与设备配置 ===
    channels = ['直通车', '自然搜索', '社交媒体']
    channel_weights = [0.58, 0.27, 0.15]

    devices = ['移动端', '桌面端']
    device_weights = [0.68, 0.32]

    # === 漏斗转化率配置 ===
    # A组（对照组）：标准图文列表
    # Overall: 0.76 × 0.38 × 0.178 ≈ 5.14%
    funnel_a = {
        'click_to_view': 0.76,
        'view_to_interact': 0.38,
        'interact_to_submit': 0.178
    }

    # B组（实验组）：交互式需求评估与动态报价器
    # Overall: 0.78 × 0.52 × 0.165 ≈ 6.69%
    funnel_b = {
        'click_to_view': 0.78,
        'view_to_interact': 0.52,
        'interact_to_submit': 0.165
    }

    # === 修正系数 ===
    channel_mod = {'直通车': 1.0, '自然搜索': 1.12, '社交媒体': 0.78}
    device_mod = {'移动端': 0.92, '桌面端': 1.10}

    # === 时段流量分布 (24h) ===
    hour_w = np.array([
        0.008, 0.004, 0.002, 0.002, 0.003, 0.008,
        0.015, 0.035, 0.065, 0.085, 0.095, 0.088,
        0.078, 0.082, 0.088, 0.085, 0.072, 0.055,
        0.042, 0.035, 0.028, 0.018, 0.012, 0.008
    ])
    hour_w = hour_w / hour_w.sum()

    # === 服务类型与价格 ===
    service_types = ['PPT定制', '代写文章', '编程代写', '网站开发', '机械设计', '数据分析']
    service_probs = [0.25, 0.20, 0.22, 0.15, 0.10, 0.08]
    price_ranges = {
        'PPT定制': (50, 300), '代写文章': (80, 500), '编程代写': (150, 800),
        '网站开发': (500, 3000), '机械设计': (200, 1200), '数据分析': (200, 1000)
    }

    records = []
    uid = 0

    for day_idx in range(n_days):
        cur_date = start + timedelta(days=day_idx)
        dow = cur_date.dayofweek
        is_weekend = dow >= 5

        # 泊松分布日流量 + 周末衰减
        n_visitors = rng.poisson(daily_traffic_lambda)
        if is_weekend:
            n_visitors = int(n_visitors * 0.82)

        # 轻微上升趋势（广告投放持续优化）
        day_trend = 1.0 + 0.002 * day_idx

        for _ in range(n_visitors):
            uid += 1
            group = rng.choice(['A', 'B'])
            channel = rng.choice(channels, p=channel_weights)
            device = rng.choice(devices, p=device_weights)
            hour = int(rng.choice(24, p=hour_w))
            funnel = funnel_a if group == 'A' else funnel_b

            # 综合修正系数
            mod = (
                channel_mod[channel]
                * device_mod[device]
                * (0.83 if is_weekend else 1.0)
                * day_trend
                * max(0.6, rng.normal(1.0, 0.08))
            )

            # ---- 漏斗模拟 ----
            stage = 1
            time_on_page = 0.0
            pages_viewed = 1

            if rng.random() < min(0.98, funnel['click_to_view'] * mod):
                stage = 2
                time_on_page = max(3.0, rng.lognormal(3.2, 0.9))
                pages_viewed = max(1, int(rng.poisson(2.5)))

                if rng.random() < min(0.95, funnel['view_to_interact'] * mod):
                    stage = 3
                    time_on_page += max(8.0, rng.lognormal(3.8, 0.7))
                    pages_viewed += max(1, int(rng.poisson(1.8)))

                    if rng.random() < min(0.90, funnel['interact_to_submit'] * mod):
                        stage = 4
                        time_on_page += max(15.0, rng.lognormal(3.0, 0.6))
                        pages_viewed += max(1, int(rng.poisson(1.2)))

            converted = 1 if stage >= 4 else 0

            # 广告成本
            ad_cost = round(rng.uniform(1.8, 3.8), 2) if channel == '直通车' else 0.0

            # 收入
            if converted:
                stype = rng.choice(service_types, p=service_probs)
                revenue = round(rng.uniform(*price_ranges[stype]), 2)
            else:
                stype = None
                revenue = 0.0

            records.append({
                'user_id': f'U{uid:06d}',
                'date': cur_date.strftime('%Y-%m-%d'),
                'hour': hour,
                'group': group,
                'channel': channel,
                'device': device,
                'stage_reached': stage,
                'time_on_page': round(time_on_page, 1),
                'pages_viewed': pages_viewed,
                'is_weekend': is_weekend,
                'day_of_week': dow,
                'converted': converted,
                'ad_cost': ad_cost,
                'revenue': revenue,
                'service_type': stype
            })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    stage_map = {1: '广告点击', 2: '浏览页面', 3: '互动/阅读', 4: '提交线索'}
    df['stage_label'] = df['stage_reached'].map(stage_map)
    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """计算各组汇总统计指标。"""
    result = {}
    for g in ['A', 'B']:
        gdf = df[df['group'] == g]
        n = len(gdf)
        conv = gdf['converted'].sum()
        rev = gdf['revenue'].sum()
        cost = gdf['ad_cost'].sum()
        result[g] = {
            'visitors': n,
            'conversions': conv,
            'conv_rate': conv / n if n else 0,
            'revenue': rev,
            'ad_cost': cost,
            'roi': rev / cost if cost else 0,
            'cpa': cost / conv if conv else 0,
            'avg_time': gdf.loc[gdf['stage_reached'] >= 2, 'time_on_page'].mean(),
            'avg_pages': gdf['pages_viewed'].mean()
        }
    return result


if __name__ == '__main__':
    df = generate_ab_test_data()
    print(f"生成数据: {len(df)} 条记录")
    print(f"日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    s = get_summary_stats(df)
    for g, v in s.items():
        print(f"\n{'对照' if g == 'A' else '实验'}组 ({g}):")
        print(f"  访客 {v['visitors']:,}  转化 {v['conversions']}  "
              f"转化率 {v['conv_rate']:.2%}  ROI 1:{v['roi']:.2f}")
