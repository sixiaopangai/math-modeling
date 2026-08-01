# 自来水厂水质预测与评估（APMCM A 题）

2026 年校级数学建模竞赛参赛项目，赛题引自 APMCM 亚太赛 A 题《自来水厂水质预测与评估》。数据为某水厂 2025-01 至 2026-03 的运行监测记录，每 2 小时一条，重建后共 5460 个时点、77 个字段。

本仓库包含从原始表格到营运风险评级的完整可复现链路：数据质控、四问建模、绘图、结果工作簿、LibreOffice 重算验证、哈希清单，以及论文 LaTeX 主稿。

## 四问与交付

| 问题 | 任务 | 主要交付 |
|---|---|---|
| Q1 | 筛选影响出厂水浊度的主要因素，建立函数关系，预测 2026-02-01/10/20 三日浊度 | 因素筛选、条件预测与 95% 区间、外部检验 |
| Q2 | 建立原水指标与操作变量到滤后水浊度的动态时滞模型 | 各输入有效时滞、ARX-Ridge 参数、RMSE/R²/残差诊断 |
| Q3 | 质量守恒（清水池 RTD）+ 数据驱动的混合模型，预测未来 1–12 h 出厂浊度 | 分时距回测、500 条联合轨迹、情景敏感性 |
| Q4 | 以 1 NTU 为硬约束，按超标幅度与持续时长建立四级风险评价 | 等级占比、逐日分级、81 组阈值稳健性 |

> **阈值口径**：1 NTU 是国标硬约束；SAFE/LOW/MEDIUM/HIGH 四级阈值是本题为联合编码「幅度 + 持续时长」而设计的运营评价尺度，**不是国家标准**，也不替代当地法规或水厂正式控制限。

## 目录结构

```
.
├── data/                          只读输入，唯一权威数据源
│   ├── A题 自来水厂水质预测与评估.pdf
│   └── 附件/{2025数据集, 2026数据集}   12 个 .xlsx + 3 个 .xls
├── src/                           全部计算代码
│   ├── water_quality_model.py     数据加载、时间轴重建、共用工具与哈希
│   ├── water_quality_full.py      四问建模主计算
│   ├── water_quality_figures.py   12 幅正式图 + 灰度审查图
│   ├── reproduce.py               全量复现入口（唯一命令）
│   ├── diagnose_shift.py          事后诊断：分布漂移与指标口径
│   ├── diagnose_figure.py         事后诊断：诊断图与误差分解
│   └── utils/plot_style.py        统一绘图样式（字体、配色、300 dpi）
├── .vendor/                       随仓库分发的 xlrd 2.0.2（读取 .xls）
├── results/                       全部数值产物（见下表）
├── figures/                       论文插图 PNG + SVG，_qa/ 为灰度审查版
├── paper/                         论文
│   ├── 完整论文.pdf               交付 PDF
│   ├── 完整论文.build.json        构建清单（绑定源码哈希与 PDF 哈希）
│   ├── 完整论文-LaTeX/            论文源码（main.tex + references.bib + figures）
│   └── paper_work/                Word 转换与渲染中间产物（不作为交付物跟踪）
├── docs/                          分析与交接文档
│   ├── 题目分析报告.md            建模前方案设计，含预先登记的判定规则
│   ├── T1诊断分析报告.md          外部验证失败的事后诊断报告
│   ├── 术语表格.md                字段与术语对照
│   └── 交接文档.md                收尾阶段任务说明
└── backups/                       论文修改前备份（不跟踪）
```

## 环境依赖

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.12 | 记录运行为 3.12.3（Linux），已在 3.12.10（Windows）复核 |
| numpy | 2.5.1 | |
| pandas | 3.0.5 | |
| scipy | 1.18.0 | |
| matplotlib | 3.11.1 | |
| openpyxl | 3.1.5 | |
| xlrd | 2.0.2 | **无需安装**，已随仓库置于 `.vendor/` |
| LibreOffice | 任意近期版本 | 提供 `soffice`，用于工作簿重算门禁 |

安装（除 xlrd 外）：

```bash
python -m pip install "numpy==2.5.1" "pandas==3.0.5" "scipy==1.18.0" "matplotlib==3.11.1" "openpyxl==3.1.5"
```

编译论文另需 XeLaTeX 与 `ctex` 宏包，且 LaTeX 内核不低于 2026/06/01（`ctex` v2.6.4 的要求）：TeX Live 2026，或已执行过 `miktex packages update` 的 MiKTeX。

## 一条命令完成全量复现

```bash
PYTHONPATH=.vendor python src/reproduce.py
```

Windows PowerShell 下等价写法：

```powershell
$env:PYTHONPATH=".vendor"; python src/reproduce.py
```

该命令依次执行：输入只读性校验 → 四问计算 → 绘图 → 图件审计（300 dpi、PNG/SVG 配对）→ 工作簿结构校验 → LibreOffice 重算 → 复算后二次校验 → 生成哈希清单。任一门禁不通过即抛出异常并停止。成功时输出 `{"status": "ok", ...}`。

**运行前提**：`data/` 下所有文件必须是只读的，否则输入保护门禁会失败。这是刻意设计，用于保证原始附件不被任何输出写入。

```bash
chmod -R a-w data
```

```powershell
Get-ChildItem data -Recurse -File | ForEach-Object { $_.IsReadOnly = $true }
```

事后诊断不属于建模主链路，在主链路跑完之后单独复现：

```bash
PYTHONPATH=.vendor python src/diagnose_shift.py
PYTHONPATH=.vendor python src/diagnose_figure.py
```

编译论文：

```bash
cd paper/完整论文-LaTeX && latexmk -norc -gg -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

> 若使用 `-outdir`，部分发行版的 BibTeX 会在输出目录中找不到 `references.bib`，导致参考文献全部为空而编译仍返回 0。此时先设置 `BIBINPUTS` 指向项目目录（Linux/macOS 用 `export BIBINPUTS="$PWD:"`，Windows 用 `$env:BIBINPUTS="$PWD;"`）。验收标准：编译日志中 `undefined` 计数为 0，PDF 为 27 页，参考文献 5 条。

## 脚本与产物对应关系

| 脚本 | 产物 |
|---|---|
| `src/water_quality_full.py --full` | `results/` 下全部 CSV/JSON/NPZ、`results/建模结果.xlsx` |
| `src/water_quality_figures.py --all` | `figures/*.png`、`figures/*.svg`、`figures/_qa/*_grayscale.png`、`results/图表契约.json` |
| `src/reproduce.py` | `results/input_protection.json`、`figure_audit.json`、`xlsx_validation.json`、`xlsx_recalc_external.json`、`复现清单.json` |
| `src/diagnose_shift.py` | `results/分布漂移诊断.csv`、`results/指标口径诊断.json` |
| `src/diagnose_figure.py` | `figures/diag_distribution_shift.png/.svg`、`results/误差分解诊断.json` |

主要结果文件：

| 文件 | 内容 |
|---|---|
| `results/建模结果.xlsx` | 题目要求的答案工作簿，17 个 sheet |
| `q1_predictions.csv` / `q1_evaluation.csv` | 二月三日条件预测与区间 / 三月外部检验 |
| `q1_factors.csv` / `q1_function_parameters.csv` / `q1_partial_effects.csv` | 因素筛选 / 60 行函数参数 / 单变量条件响应 |
| `q2_delays.csv` / `q2_accuracy.csv` / `q2_ar_order_sensitivity.csv` | 有效时滞与可辨识性 / 拟合精度与残差诊断 / AR 阶次敏感性 |
| `q3_designated_predictions.csv` / `q3_horizon_accuracy.csv` / `q3_sensitivity.csv` | 指定日 1–12 h 预测 / 分时距回测 / 情景敏感性 |
| `q3_uncertainty_500_paths.npz` | 500 条联合递推轨迹，数组维度 500×5460 |
| `q4_daily.csv` / `q4_summary.csv` | 逐营运日指标与等级 / 等级占比 |
| `results/复现清单.json` | 绑定输入哈希、代码哈希、产物哈希、随机种子与关键参数 |
| `results/数据源等价性判定.json` | 曾并存的第二份附件与 `data/附件/` 的逐文件语义哈希核对记录 |

## 复现验证顺序

复现完成后按此顺序核对：数据行数与时间轴（5460 个时点、336 个二月目标缺失）→ 四问核心数值 → 工作簿公式错误为 0 → 图表契约 → 文件哈希。全部与 `results/复现清单.json` 一致时，才视为同一证据版本。

跨平台运行会引入 1e-13 量级的浮点差异，`plot_*` 辅助文件与 npz 数组的字节哈希因此可能变化；论文引用的全部 CSV 在 Linux 与 Windows 上均逐字节一致。

## 方法学约束

以下约束在建模前预先登记（见 `docs/题目分析报告.md`），复现与二次开发时不得突破：

1. 2026 年 3 月数据仅用于最终外部报告，**不参与任何超参数选择或模型筛选**。
2. 时间划分固定：训练/开发 = 2025-01 至 2026-01；最终预测 = 2026-02（出厂浊度目标整段缺失）；外部验证 = 2026-03。
3. 每问模型体系上限 2 套。
4. 随机种子固定 `20260727`。
5. 原始附件只读，任何输出不得写入 `data/`。
6. 相关性表述不得写成因果性；条件效应只代表现有运行策略下的统计关联。
7. `docs/T1诊断分析报告.md` 中问题一的 −0.3433 NTU 系统偏差是**诊断结论，不是可回填进模型的参数**——它由三月实测算得，且二月目标整段缺失时该校正本就无法实施。

## 已知限制

问题一的两套模型在 2026 年 3 月外部检验中均弱于「同营运时次前一日」基线（R² 分别为 −3.447 和 −2.995）。误差分解表明其中 76.64% 来自系统性水平偏移而非形态失配，因此结论表述为「水平定标失效、相对形态可用」。仓库刻意保留这些负面证据，不做粉饰。
