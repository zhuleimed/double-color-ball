# CLAUDE.md

双色球 LSTM-Transformer 智能预测管线项目。全自动完成数据同步、模型预测、开奖结果对比和微信推送。

## 项目概览

- **目标**: 利用深度学习预测双色球号码，追踪累积成功率
- **服务器**: 36核 Intel Xeon Gold 6140 @ 2.30GHz, 187GB RAM, 无GPU，本项目限用33核
- **Python**: `/home/zhulei/anaconda3/bin/python` (3.12.7)
- **GitHub**: `git@github.com:zhuleimed/DoubleColorBall.git`
- **完整指南**: `双色球预测管线项目指南.docx`

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
├── ssq_model/               # 模型预测模块
│   ├── config.py            # ModelConfig: window=90, red_classes=33, optuna_n_trials=50
│   ├── features.py          # 增强特征: 红球82维/期(频次+遗漏+区间+和值+AC值等)
│   ├── red_model.py         # LSTM-Transformer 6×33分类 + Beam Search约束解码
│   │                        #   早停: patience=20 + ReduceLROnPlateau
│   ├── blue_model.py        # LGB+XGB+CatBoost+RF → Stacking(LR) 16分类
│   ├── ensemble.py          # SSQEnsemble: 加载模型→预测→保存
│   ├── train.py             # 训练入口 (--full/--quick/--red-only/--blue-only)
│   └── predict.py           # 预测入口 (--beam/--no-save/--verbose)
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

# 训练
python -m ssq_model.train --quick                # 快速训练(默认参数)
python -m ssq_model.train --full                 # 完整训练(Optuna 50 trials)
python -m ssq_model.train --red-only --quick     # 仅红球快速训练
python -m ssq_model.train --blue-only --quick    # 仅蓝球快速训练

# 预测
python -m ssq_model.predict --verbose            # 预测下期(详细输出)

# 管线
python pipeline.py                               # 完整管线(同步→对比→预测→推送)
python pipeline.py --predict-only                # 仅预测+推送
python pipeline.py --backfill                    # 全量回填
```

## Cron 调度

```
# 每日管线（周三/五/一 10:00，开奖日次日）
0 10 * * 3,5,1 cd ... && python pipeline.py >> logs/pipeline_$(date +\%Y\%m\%d).log

# 月度完整训练（月末倒数第二天 00:00，提前 ~34h 确保管线日前完成）
# 先同步最新数据，再训练。训练约 30-35h，为管线日 10:00 留足缓冲。
0 0 27-31 * * [ "$(date -d '+2 day' +\%d)" = "01" ] && cd ... && python -m ssq_sync.main --sync-only && python -m ssq_model.train --full >> logs/retrain_$(date +\%Y\%m).log
```

**为什么是月末倒数第二天？** 完整训练实测需 30h+，若在 1 号 00:00 启动，当天 10:00 管线来不及。提前 1.5 天启动后，管线日永远有新模型可用。训练前先 `--sync-only` 确保数据最新，缺口仅 1-2 期（0.05%）可忽略。

## 模型架构

- **红球**: LSTM(128)→TransformerBlock×2→LSTM(64)→6×Dense(33,softmax)
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
