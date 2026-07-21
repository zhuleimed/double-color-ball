# CLAUDE.md

双色球统计分析管线项目。全自动完成数据同步、统计偏差检测、组合覆盖策略分析、蓝球统计分析、开奖结果对比和微信推送。

> **v2.1 更新 (2026-07-21):** 模型预测模块(ssq_model)已停用，替换为统计分析模块(ssq_analysis)，
>   包含三项分析: 物理偏差检测 + 组合覆盖策略 + 蓝球统计分析。
>   月度模型训练cron已暂停。原ssq_model代码保留但不再执行。

## 项目概览

- **目标**: 通过统计分析方法研究双色球号码规律，提供偏差检测、轮选策略和蓝球分析
- **服务器**: 36核 Intel Xeon Gold 6140 @ 2.30GHz, 187GB RAM, 无GPU
- **Python**: `/home/zhulei/anaconda3/bin/python` (3.12.7)
- **GitHub**: `git@github.com:zhuleimed/DoubleColorBall.git`

## 业务节奏

双色球每周二/四/日 21:15 开奖，管线每日 10:00 执行（周三/五/一跑完整管线）。

## 项目结构

```
DoubleColorBall/
├── pipeline.py              # 管线编排器（默认模式=完整管线）
├── pipeline_status.py       # 状态追踪（JSON原子持久化）
├── .env                     # WxPusher Token配置（不提交Git）
│
├── ssq_sync/                # 数据同步模块
│   ├── config.py / logger.py
│   ├── data_source.py       # cwl.gov.cn官方API → zhcw.com爬虫
│   ├── engine.py            # SQLite引擎（draw_history, sync_log, prediction_log, result_compare）
│   ├── sync.py              # SSQSync: 全量回填 + 增量同步 + 开奖日判断
│   ├── notify.py / main.py
│
├── ssq_model/               # 模型预测模块 (已停用，代码保留)
│   ├── config.py / features.py / red_model.py / blue_model.py
│   ├── ensemble.py / train.py / predict.py
│
├── ssq_analysis/             # 统计分析模块 (当前使用)
│   └── main.py               # 三合一分析: 偏差检测+覆盖策略+蓝球分析
│
├── ssq_report/              # 结果对比与报告模块
│   ├── compare.py           # 预测vs实际: 命中数+中奖等级判定
│   ├── tracker.py           # SuccessTracker: 累积成功率JSON持久化+滚动窗口
│   ├── reporter.py          # 日报文本生成
│   └── notify.py            # WxPusher推送
│
├── data/ (gitignore)        # ssq_history.db (2039期) + models/
├── logs/ (gitignore)        # pipeline_YYYYMMDD.log
└── output/ (gitignore)      # pipeline_status.json + success_tracker.json
```

## 常用命令

```bash
# 数据
python -m ssq_sync.main --backfill              # 全量回填
python -m ssq_sync.main --sync-only              # 增量同步

# 统计分析
python -m ssq_analysis.main                      # 运行三项分析(偏差+覆盖+蓝球)

# 管线
python pipeline.py                               # 完整管线(同步→对比→分析→推送)
python pipeline.py --backfill                    # 全量回填

# 模型训练(已停用，代码保留)
# python -m ssq_model.train --full               # 月度完整训练(Optuna 50 trials)
```

## Cron 调度

```
# 每日管线（周三/五/一 10:00，开奖日次日）
0 10 * * 3,5,1 cd ... && python pipeline.py >> logs/pipeline_$(date +\%Y\%m\%d).log

# 月度模型训练 — 已暂停 (2026-07-21)。模型预测替换为统计分析。
# 如需恢复，取消下面注释:
# 0 0 27-31 * * [ "$(date -d '+2 day' +\%d)" = "01" ] && cd ... && python -m ssq_sync.main --sync-only && python -m ssq_model.train --full >> logs/retrain_$(date +\%Y\%m).log
```

## 模型架构（已停用，仅供参考）

- **红球**: LSTM(128)→TransformerBlock×3→LSTM(64)→6×Dense(33,softmax)
  - Beam Search约束解码保证递增+不重复
  - 早停 patience=20, ReduceLROnPlateau factor=0.5
- **蓝球**: LightGBM+XGBoost+CatBoost+RF → Stacking(LogisticRegression)
  - 5折交叉验证生成元特征
- **Optuna**: 50 trials, n_jobs=4, TF_NUM_INTRAOP/INTEROP_THREADS=8, OMP/MKL等=4
  - 峰值CPU: 4×8=32核（在33核限制内）
- **特征**: 红球82维/期(频次33+遗漏33+号码6+区间3+和值/跨度/AC值3+奇偶/大小/质合比3+连号1)
- **训练日志**: 详细阶段性输出（TrainingLogger回调: 每5轮准确率/损失/lr, 过拟合检测, 早停原因；Optuna TrialProgressCallback: 实时进度/最佳值/ETA；蓝球逐折准确率）

## 训练日志说明

日志中关键标记:
- `★` — 当前最佳轮次（val_loss 新低）
- `⚠ 过拟合` — train_loss 和 val_loss 差距 > 0.5，模型可能过拟合
- `[T{trial}]` — Optuna 第 trial 号试验
- `[最终训练]` — 用最佳参数进行的最终完整训练
- `[蓝球]` — 蓝球模型训练阶段
- `[蓝球Optuna]` — 蓝球超参数搜索
- `[Optuna] Trial N/M 完成` — 搜索进度，含全局最佳值和预计剩余时间

## 数据库表

- `draw_history`: 开奖历史(period, draw_date, red1-6, blue)
- `sync_log`: 同步日志
- `prediction_log`: 预测记录(period, pred_date, red1-6, blue, model_version)
- `result_compare`: 结果对比(period, pred_red_hits, pred_blue_hit, hit_details, prize_level)

## 关键依赖

tensorflow-cpu, xgboost, lightgbm, catboost, optuna, scikeras, scikit-learn,
pydantic-settings, python-dotenv, rich, wxpusher, requests, beautifulsoup4, lxml
