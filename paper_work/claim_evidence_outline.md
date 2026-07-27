# W1 主张—证据大纲（内部检查）

## 0. 范围与格式来源

- 论文题目：自来水厂水质预测、动态传播与风险评估
- 语言：中文。
- 格式：用户明确授权使用 Skill 默认模板；采用内置 `cumcm` 构建基线，但不声称符合本年度实际竞赛模板。
- 共同证据源：`results/`、`figures/`、`water_quality_full.py`；Word 与 LaTeX 不分别改写数值或结论。
- 质量目标：约 20 页，至少 8 幅正式图、3 个表、5 个行间公式；计划使用 12 幅正式图。
- 流程偏离：修复后 P2 作者侧复现、Calc、哈希和图审计通过，但独立复验因协作通道故障未取得 PASS；用户明确要求进入论文阶段。

## 1. 摘要拟用主张

| 主张 | 精确数值 | 证据 |
|---|---:|---|
| 数据形成无重复的两小时时间轴 | 5460 行、5460 个唯一时戳；5 个时间单元修复；2026-02 出厂水 NTU 缺失 336/336 | `results/full_summary.json:data`；`results/data_quality.csv`；`water_quality_model.py` 数据读取链 |
| Q1 稳健因素筛选 | `FILT. NTU` 和 `R/W CLR`；置换 RMSE 增量分别 0.115354、0.024035 NTU；四折方向一致率均 1.00 | `results/q1_factors.csv` 第 1、2 行；`figures/process_q1_factor_screening.png` |
| Q1 不具外部泛化证据 | Huber-Ridge RMSE 0.392112、R2 -3.447487；RFF RMSE 0.371614、R2 -2.994651；前一营运日基线 RMSE 0.170873 | `results/q1_evaluation.csv`；`figures/result_q1_february_predictions.png`；条件预测必须标注为条件估计 |
| Q2 最优动态结构 | 时滞依次为 2、0、0、6 h；岭 0.01；AR 阶 6 | `results/q2_delays.csv`；`results/plot_q2_search.csv` 1280 行；`figures/process_q2_lag_search.png` |
| Q2 外测和残差限制 | 2026-03 RMSE 0.166450、R2 -0.373590；1--6 阶残差最大绝对 ACF 0.416470 | `results/q2_accuracy.csv`；`figures/result_q2_external_validation.png` |
| Q3 物理—数据混合预测 | RTD 时间尺度 8 h，核质量 1.0；500 条联合递推轨迹 | `results/full_summary.json:q3`；`results/q3_uncertainty_500_paths.npz`；`figures/process_q3_rtd_selection.png` |
| Q3 分时距能力 | 混合模型 2/4/6/8/10/12 h RMSE 为 0.09703/0.13668/0.15091/0.14201/0.12537/0.13904；仅 12 h 优于持续性；95% 覆盖率 0.967--0.978 | `results/q3_horizon_accuracy.csv`；`figures/result_q3_horizon_accuracy.png` |
| Q4 营运日等级 | 90 天中 SAFE/LOW/MEDIUM/HIGH = 86/2/1/1；占比 95.56%/2.22%/1.11%/1.11% | `results/q4_summary.csv`；`figures/result_q4_daily_grades.png` |
| Q4 三月异常日 | 2026-03-12 MEDIUM、2026-03-13 LOW | `results/q4_daily.csv` 对应行；不得遗漏二月区间上界可能扩展至 MEDIUM/HIGH |

## 2. 章节与证据映射

### 2.1 数据、预处理和共用验证设计

- 主张：时间轴严格按营运日 07:00 至次日 05:00、步长 2 h 重建，不用自然日错位；数值字段保留 `raw/clean/invalid` 三层。
- 表：表 1 数据字段、单位与缺失概况，来源 `results/data_quality.csv`。
- 图：图 1 `figures/raw_q1_target_missing.png`，说明二月目标完整缺失；正文标签 `fig:q1-missing`。
- 公式：对数变换、缩尾和标准化定义。
- 代码：`water_quality_model.py` 的读取、修复、时间轴断言；`results/复现清单.json` 的输入哈希。
- 写作限制：5 个 `5.87 -> 1700` 是可审计时间单元修复，不写成数值异常插补。

### 2.2 问题一：因素筛选、函数关系和条件预测

- 模型：Huber 加权岭回归为解释主模型，RFF-Ridge 为非线性对照；两者属于两套模型体系。
- 核心公式：
  1. `z_tj=[clip(g_j(x_tj),l_j,u_j)-mu_j]/s_j`；
  2. `log(1+y_t)=beta_0+sum_j beta_j z_tj+epsilon_t`；
  3. Huber 权重 `w=1`（`|u|<=1.345`）或 `1.345/|u|`；
  4. 置换贡献 `Delta_j=RMSE_perm,j-RMSE_base`。
- 函数参数：表 2 摘列截距和两项主要因素的当前/滞后系数；完整 60 行参数见 `results/q1_function_parameters.csv`，偏效应见 `results/q1_partial_effects.csv`。
- 因素结论：仅 `filt_ntu` 与 `rw_clr` 同时满足置换贡献为正、排名前 6、四折方向一致率至少 75%；其方向分别为正、负。仅解释关联，不写因果。
- 图：
  - 图 2 `figures/process_q1_factor_screening.png`，`fig:q1-factor`；
  - 图 3 `figures/result_q1_february_predictions.png`，`fig:q1-forecast`。
- 预测：36 个指定营运时点及 95% 区间来自 `results/q1_predictions.csv`。
- 验证：表 3 使用 `results/q1_evaluation.csv`；外部失败必须紧随预测说明，不能把二月条件估计写成可靠外推。
- 代码：`water_quality_full.py:161` 起的 Q1 训练、滚动验证、置换和函数表输出。
- 文献落点：Kim & Parnichkun (2017) 说明水厂浊度与投药预测背景；Yasin et al. (2021) 支撑稳健岭 M 估计；Avron et al. (2017) 支撑 RFF 核岭对照；Chicco et al. (2021) 支撑同时报告 RMSE 与 R2。

### 2.3 问题二：滤后水动态响应与时滞

- 模型：带输入平方项、交互项与滤后水自回归项的 ARX-Ridge。
- 核心公式：`log(1+F_t)=a_0+sum_{k=1}^p phi_k log(1+F_{t-k})+sum_m h_m(X_{m,t-d_m})+epsilon_t`。
- 枚举：4 个输入各取 0/2/4/6 h，形成 256 个组合；每组 5 个岭值，共 1280 次滚动验证。
- 表：表 4 来源 `results/q2_delays.csv` 和 `results/q2_accuracy.csv`，给出时滞、交叉相关检查、外测误差和残差诊断。
- 图：
  - 图 4 `figures/raw_q2_input_output_series.png`，`fig:q2-series`；
  - 图 5 `figures/process_q2_lag_search.png`，`fig:q2-lag`；
  - 图 6 `figures/result_q2_external_validation.png`，`fig:q2-validation`。
- 结论边界：四个时滞可作为数据支持的有效延迟，不解释为唯一水力停留时间；外测 R2 为负且 ACF 仍高，说明尖峰和剩余依赖未被模型充分解释。
- 代码：`water_quality_full.py` 中 `q2_design_order`、`run_q2`。
- 文献落点：Chen et al. (2020) 用于水质时间序列模型选择背景；不以文献替代本数据的时滞结论。

### 2.4 问题三：清水池传播、1--12 h 预测与不确定性

- 模型：有限质量归一化的指数 RTD 物理基线 + 对数残差 Huber-Ridge；移动块残差产生 500 条联合递推轨迹。
- 核心公式：
  1. `q=exp(-Delta t/tau)`，`k_j=q^j/sum_{r=0}^{J-1}q^r`；
  2. `P_t=sum_j k_j F_{t-j}`；
  3. `log(1+Y_{t+h})=log(1+P_{t+h})+f_h(S_t)+epsilon_{t,h}`。
- 参数：`tau=8 h`，核质量为 1；奇数小时只作相邻偶数小时对数尺度插值，不能当作独立训练模型。
- 表：表 5 来源 `results/q3_horizon_accuracy.csv`；指定日预测取 `results/q3_designated_predictions.csv` 和 `results/q3_horizon_predictions.csv`。
- 图：
  - 图 7 `figures/raw_q3_filtered_treated_series.png`，`fig:q3-propagation`；
  - 图 8 `figures/process_q3_rtd_selection.png`，`fig:q3-rtd`；
  - 图 9 `figures/result_q3_horizon_accuracy.png`，`fig:q3-accuracy`。
- 敏感性：从 `results/q3_sensitivity.csv` 报告场景变化小且方向随时距变化，不概括为稳定因果效应。
- 验证边界：物理基线单独误差大；混合模型仅 12 h 略优于持续性；高覆盖率同时反映区间较宽。
- 代码：`water_quality_full.py` 中 RTD 选择、`moving_block_errors`、500 路径递推和分时距回测。
- 文献落点：Toson et al. (2019) 支撑指数 RTD 是 CSTR 的基本分布及其平滑解释；Ebtehaj et al. (2010) 支撑移动块自助法保留时间相关性。

### 2.5 问题四：营运日风险分级

- 定义：`e_t=max(Y_t-1,0)`，最大超标幅度 `A=max e_t`，最长超标时长 `D=2*max_run(e_t>0)`，超标负荷 `L=2*sum e_t`。
- 分级：SAFE 为 `A=0`；LOW 为 `A<=0.5` 且 `D<=4 h`；MEDIUM 为 `A<=1.0` 且 `D<=8 h`；其他为 HIGH。阈值是本模型的可解释设计，不伪称法规规定。
- 不确定性：二月对 500 条轨迹逐日评级，报告众数、概率及 95% 等级范围；每日有效点少于 9 个时不能直接评级。
- 表：表 6 来源 `results/q4_summary.csv` 和 `results/q4_daily.csv`；阈值稳健性来源结果工作簿 `Q4稳健性`。
- 图：
  - 图 10 `figures/raw_q4_daily_maximum.png`，`fig:q4-maximum`；
  - 图 11 `figures/process_q4_risk_matrix.png`，`fig:q4-matrix`；
  - 图 12 `figures/result_q4_daily_grades.png`，`fig:q4-grades`。
- 代码：`water_quality_full.py` 中 `classify_custom`、`run_q4`；基础评级合同在 `water_quality_model.py`。
- 结论边界：SAFE 占比高不等于二月确定安全；二月中心等级均为 SAFE，但多日区间上界可至 MEDIUM/HIGH。

## 3. 模型评价与适用边界

- 优点：营运日时间轴明确；物理传播和数据残差分离；外部月份验证；所有不确定性以联合轨迹传递到风险等级；完整哈希可复现。
- 局限：二月目标完全缺失，无同期实测校准；Q1/Q2 外测 R2 为负；Q2 残差自相关明显；Q3 只有最长时距优于持续性；风险阈值是决策设计而非监管标准。
- 改进：增加在线浊度传感器质控、清水池水力参数和事件标签；采用变参数/状态空间模型；积累二月或跨年度外测数据后重估区间。

## 4. 已核验参考文献

1. Kim, C. M.; Parnichkun, M. Prediction of settled water turbidity and optimal coagulant dosage... Applied Water Science, 2017, 7(7): 3885--3902. DOI: 10.1007/s13201-017-0541-5. OpenAlex/AnySearch 交叉匹配；Springer 页面 HTTP 200。
2. Chen, Y. et al. A Review of the Artificial Neural Network Models for Water Quality Prediction. Applied Sciences, 2020, 10(17): 5776. DOI: 10.3390/app10175776. 双引擎交叉匹配；DOI 已解析至 MDPI；Crossref 元数据核对。
3. Yasin, S. et al. Modified Robust Ridge M-Estimators in Two-Parameter Ridge Regression Model. Mathematical Problems in Engineering, 2021: 1--24. DOI: 10.1155/2021/1845914. 双引擎交叉匹配；Crossref 元数据核对。
4. Avron, H. et al. Random Fourier Features for Kernel Ridge Regression: Approximation Bounds and Statistical Guarantees. PMLR 70, 2017: 253--262. 出版机构页面 HTTP 200，citation meta 已核对；OpenAlex/AnySearch 交叉匹配。
5. Toson, P.; Doshi, P.; Jajcevic, D. Explicit Residence Time Distribution... Processes, 2019, 7(9): 615. DOI: 10.3390/pr7090615. 双引擎交叉匹配；Crossref 元数据核对。
6. Ebtehaj, M.; Moradkhani, H.; Gupta, H. V. Improving robustness of hydrologic parameter estimation by the use of moving block bootstrap resampling. Water Resources Research, 2010, 46(7). DOI: 10.1029/2009WR007981. 双引擎交叉匹配；Crossref 元数据核对。
7. Chicco, D.; Warrens, M. J.; Jurman, G. The coefficient of determination R-squared is more informative... PeerJ Computer Science, 2021, 7: e623. DOI: 10.7717/peerj-cs.623. 双引擎交叉匹配；Crossref 元数据核对。
8. Hampel, F. R. Introduction to Huber (1964) Robust Estimation of a Location Parameter. In: Breakthroughs in Statistics, 1992: 479--491. DOI: 10.1007/978-1-4612-4380-9_34. 双引擎交叉匹配；Springer 页面 HTTP 200。

## 5. W1 核验项

- [x] q1--q4 均有公式、结果表、正式图、代码位置和限制。
- [x] 摘要的全部数值可在 CSV/JSON 精确定位。
- [x] 12 幅图均有连续编号、问题标签和正文落点。
- [x] Word 与 LaTeX 使用同一主稿、表格、图片和参考文献。
- [x] 文献经 OpenAlex/AnySearch 交叉匹配，并打开 DOI/出版页面核验。
- [x] 不把失败外测、插值时距、风险阈值或关联方向夸大为泛化、独立模型、法规或因果。
