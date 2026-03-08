# 🔬 数字化服务工作室转化链路与动态报价 A/B 测试平台

> **北大的小码农** | Monte Carlo Simulation × Statistical Inference × Bayesian Analysis

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com/)

## 🌐 在线演示

**👉 [https://yunxiaohall.com/ABTEST](https://yunxiaohall.com/ABTEST)**

---

## 📋 项目简介

本项目是一个完整的 **A/B 测试与产品策略优化** 案例，基于真实的创业运营经验，展示了从业务理解、实验设计、蒙特卡洛数据模拟、统计推断到商业洞察的全链路数据分析能力。

### 业务场景

**北大的小码农** 是一家数字化服务工作室，通过淘宝直通车投放广告，已跑通 **1:4 的广告投入回报比 (ROI)**。但发现大量客户在浏览后因"技术服务列表过长、定价不透明"而流失。

### A/B 测试设计

| 分组 | 方案 | 目标 |
|------|------|------|
| **A 组 (对照)** | 标准图文服务列表 | 基准表现 |
| **B 组 (实验)** | 交互式需求评估与动态报价器 | 降低认知门槛，提升转化 |

### 北极星指标

- **主要指标：** 表单提交转化率 (Conversion Rate)
- **护栏指标：** 线索获取成本 (CPA)、广告 ROI ≥ 1:4

---

## 🎯 项目亮点

- ✅ **蒙特卡洛模拟** — 基于泊松分布、对数正态分布模拟带有季节性波动和多维度噪音的真实日志数据
- ✅ **累积 P 值收敛图** — 展示对 A/B 测试「偷窥陷阱」(Peeking Problem) 的深入理解
- ✅ **贝叶斯推断** — Beta-Binomial 共轭模型，给出 P(B>A) 后验概率和提升幅度可信区间
- ✅ **多维分群洞察** — 识别渠道、设备、时段的差异化处理效应 (HTE)
- ✅ **商业闭环** — ROI 驱动的决策框架，从数据到业务建议

---

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 数据模拟 | Python, NumPy (蒙特卡洛) |
| 数据处理 | Pandas |
| 统计推断 | SciPy (χ², Z-test, Power Analysis) |
| 贝叶斯分析 | NumPy (Beta-Binomial 共轭) |
| 可视化 | Plotly (交互式图表) |
| Web 框架 | Streamlit |
| 部署 | Linux Server + Nginx + Systemd |

---

## 📊 仪表盘模块

1. **📊 实验概览** — 累积 P 值收敛图、每日转化趋势、流量热力图
2. **🔄 转化漏斗** — 两组用户漏斗对比、各环节转化率分析
3. **💰 ROI 分析** — 广告 ROI、CPA、收入对比、收入构成旭日图
4. **🧪 统计检验** — χ² 检验、Z 检验、置信区间、功效分析
5. **🎯 贝叶斯推断** — 后验分布、P(B>A)、相对提升分布
6. **🔍 分群洞察** — 按渠道/设备/时段的异质性分析
7. **📖 方法论** — 完整的实验设计与数据生成文档

---

## 🚀 本地运行

```bash
# 克隆仓库
git clone https://github.com/ucarcompany/-A-B-.git
cd -A-B-

# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run app.py
```

---

## 📁 项目结构

```
├── app.py               # Streamlit 主应用 (可视化仪表盘)
├── data_generator.py    # 蒙特卡洛数据模拟引擎
├── requirements.txt     # Python 依赖
├── README.md            # 项目文档
├── .gitignore
└── .streamlit/
    └── config.toml      # Streamlit 配置与主题
```

---

## 👨‍💻 关于作者

**北大的小码农** — 结合真实创业运营经验与数据分析方法论，力求展现从业务理解、实验设计、统计推断到商业洞察的全链路数据分析能力。

- GitHub: [@ucarcompany](https://github.com/ucarcompany)
- 工作室: 北大的小码农

---

*本项目数据基于真实商业经验，采用蒙特卡洛方法模拟生成，包含季节性波动与随机噪音。*
