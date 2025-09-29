# PerturbEval

轻量级扰动评估基准（S×R 闭环）。输入任务 + 题库 → 一键跑评测 → 导出指标和结果图。

## Quick Start

```bash
# 安装依赖
pip install -r requirements.txt

# 跑评测（读取 run.yaml）
python runners/run_eval.py --config run.yaml

# 从 Excel/CSV 转换指标
python scripts/tabular2metrics.py --in results/reports/run.xlsx --out results/reports/metrics.json

# 生成图表（输出到 assets/）
python scripts/plot.py --metrics results/reports/metrics.json --outdir assets --title "PerturbEval (demo)"
```

## 目录结构
```
PerturbEval/
├─ adapters/           # 模型调用
├─ data/               # 题库与答案
├─ results/
│  ├─ logs/            # 运行日志
│  ├─ preds/           # 模型预测
│  └─ reports/         # Excel/metrics.json
├─ scripts/            # 辅助脚本
│  ├─ tabular2metrics.py
│  └─ plot.py
├─ assets/             # 输出图表
├─ runners/            # 主入口 run_eval.py
├─ scoring/            # 判题与指标
├─ run.yaml            # 配置文件
├─ requirements.txt
├─ LICENSE
└─ README.md
```

## 示例结果
<img width="1080" height="720" alt="perf_by_prompt" src="https://github.com/user-attachments/assets/30e07b31-c859-43de-b14a-7dfc430c4020" />
<img width="1080" height="720" alt="robustness_quadrant" src="https://github.com/user-attachments/assets/0bd5028a-078a-4a87-a484-38090aa27552" />


## License
MIT © 2025 Maru979182063


