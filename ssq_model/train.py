"""模型训练入口。

提供 CLI 接口用于训练红球模型、蓝球模型或全部模型。
支持 Optuna 超参数搜索和最终模型训练。

运行模式:
  python -m ssq_model.train --full            # 全量训练（Optuna搜索 + 最终训练）
  python -m ssq_model.train --red-only        # 仅训练红球模型
  python -m ssq_model.train --blue-only       # 仅训练蓝球模型
  python -m ssq_model.train --quick            # 快速训练（跳过Optuna，用默认参数）
  python -m ssq_model.train --trials 10       # 指定Optuna试验次数
  python -m ssq_model.train --timeout 3600    # 指定搜索超时（秒）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ════════════════════════════════════════════════════════════
#  CPU 核心限制 — 本项目最多使用 33 核（总 36 核，预留 3 核）
#  必须在任何 import numpy/tensorflow 之前设置
# ════════════════════════════════════════════════════════════
_NUM_THREADS = "4"  # 每个并行任务最多使用 4 线程
os.environ.setdefault("OMP_NUM_THREADS", _NUM_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NUM_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _NUM_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _NUM_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _NUM_THREADS)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from datetime import datetime
from pathlib import Path

# ── 确保项目根目录在 path 中 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from dotenv import load_dotenv
load_dotenv()

from ssq_model.config import ModelConfig, get_config
from ssq_model.features import build_all_features
from ssq_model.red_model import (
    create_red_model,
    save_red_model,
    search_hyperparams as search_red,
    train_red_model,
)
from ssq_model.blue_model import (
    BlueStackingModel,
    search_blue_hyperparams,
    train_blue_model,
)
from ssq_sync.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="🔴 双色球模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m ssq_model.train --full              # 全量训练
  python -m ssq_model.train --red-only          # 仅红球模型
  python -m ssq_model.train --quick             # 快速模式（跳过搜索）
  python -m ssq_model.train --trials 30          # 指定搜索试验数
        """,
    )
    parser.add_argument("--full", action="store_true",
                        help="全量训练（Optuna搜索 + 最终训练）")
    parser.add_argument("--red-only", action="store_true",
                        help="仅训练红球 LSTM-Transformer 模型")
    parser.add_argument("--blue-only", action="store_true",
                        help="仅训练蓝球 Stacking 集成模型")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：跳过 Optuna 搜索，使用默认参数")
    parser.add_argument("--trials", type=int, default=None,
                        help="Optuna 试验次数（默认使用 config 中的值）")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Optuna 搜索超时秒数")
    parser.add_argument("--epochs", type=int, default=200,
                        help="最终训练最大轮数（默认 200）")
    args = parser.parse_args()

    config = get_config()
    t0_total = time.time()

    if not any([args.full, args.red_only, args.blue_only]):
        parser.print_help()
        print("\n请指定训练模式：--full / --red-only / --blue-only / --quick")
        sys.exit(1)

    # ── 加载特征 ──
    logger.info("=" * 55)
    logger.info(f"双色球模型训练 | {datetime.now():%Y-%m-%d %H:%M}")
    logger.info("=" * 55)

    logger.info("加载数据库并构建特征...")
    t_feat = time.time()
    data = build_all_features(config)

    red_X = data["red"]["X"]
    red_y = data["red"]["y"]
    blue_X = data["blue"]["X"]
    blue_y = data["blue"]["y"]

    logger.info("─" * 55)
    logger.info(f"📊 数据加载完成 (耗时 {time.time()-t_feat:.0f}s):")
    logger.info(f"  红球特征: X={red_X.shape} (样本×窗口×特征) y={red_y.shape}")
    logger.info(f"    → 总样本={red_X.shape[0]}, 窗口={config.window}期, 特征/期={red_X.shape[2]}维")
    logger.info(f"    → 标签范围=[{red_y.min()},{red_y.max()}], 各位置分布:")
    for i in range(3):
        logger.info(f"      位置{i+1}: 最频={np.argmax(np.bincount(red_y[:,i]))+1:02d}")
    logger.info(f"  蓝球特征: X={blue_X.shape} (样本×特征) y={blue_y.shape}")
    logger.info(f"    → 总样本={blue_X.shape[0]}, 特征={blue_X.shape[1]}维")
    logger.info(f"    → 标签分布: {np.bincount(blue_y).tolist()}")
    logger.info("─" * 55)

    version = datetime.now().strftime("%Y%m%d_%H%M")

    # ════════════════════════════════════════════════════════
    #  训练红球模型
    # ════════════════════════════════════════════════════════
    if args.full or args.red_only:
        logger.info("\n" + "=" * 55)
        logger.info("  红球 LSTM-Transformer 模型训练")
        logger.info("=" * 55)

        if args.quick:
            # 快速模式：使用默认参数
            logger.info("快速模式：跳过 Optuna 搜索")
            best_params = {
                "lstm_units": 128,
                "lstm_units2": 64,
                "num_heads": 4,
                "ff_dim": 256,
                "num_transformers": 2,
                "dropout_rate": 0.1,
                "dense_units": 128,
                "optimizer": "adam",
                "learning_rate": 0.001,
            }
        else:
            # Optuna 搜索
            logger.info("=" * 55)
            logger.info(f"Phase 1/2: Optuna 红球超参数搜索")
            logger.info(f"  搜索空间: lstm_units=32-320, num_heads=2-12, ff_dim=64-768")
            logger.info(f"  optimizer=[adam/nadam/rmsprop], lr=1e-6~3e-2")
            logger.info(f"  trials={args.trials or config.optuna_n_trials}, "
                        f"并行={config.optuna_n_jobs}, 超时={args.timeout or config.optuna_timeout}s")
            logger.info("=" * 55)
            t0 = time.time()
            search_result = search_red(
                red_X, red_y, config,
                n_trials=args.trials or config.optuna_n_trials,
                timeout=args.timeout or config.optuna_timeout,
            )
            best_params = search_result["params"]
            elapsed = time.time() - t0
            logger.info(f"[Phase 1] 搜索完成 | 耗时={elapsed:.0f}s ({elapsed/60:.1f}min)")
            logger.info(f"[Phase 1] 最佳损失: {search_result['best_value']:.6f}")
            logger.info(f"[Phase 1] 最佳参数: {json.dumps(best_params, indent=2, ensure_ascii=False)}")

        # 最终训练
        logger.info("=" * 55)
        logger.info(f"Phase 2/2: 训练最终红球模型 (epochs={args.epochs})")
        logger.info("=" * 55)
        t0 = time.time()
        red_model, red_history = train_red_model(
            red_X, red_y, best_params, config, epochs=args.epochs,
        )
        logger.info(f"红球训练完成，耗时 {time.time() - t0:.0f}s")

        # 保存模型
        save_dir = save_red_model(red_model, config, version=version)

        # 保存参数
        params_path = Path(save_dir) / "params.json"
        params_path.write_text(
            json.dumps({"version": version, "params": best_params,
                        "n_features": int(red_X.shape[2]),
                        "window": config.window}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"参数已保存: {params_path}")

        del red_model

    # ════════════════════════════════════════════════════════
    #  训练蓝球模型
    # ════════════════════════════════════════════════════════
    if args.full or args.blue_only:
        logger.info("\n" + "=" * 55)
        logger.info("  蓝球 Stacking 集成模型训练")
        logger.info("=" * 55)

        blue_params = None
        if not args.quick:
            logger.info("Phase 1: Optuna 超参数搜索...")
            t0 = time.time()
            blue_params = search_blue_hyperparams(
                blue_X, blue_y, config,
                n_trials=args.trials or min(config.optuna_n_trials, 20),
                timeout=args.timeout or config.optuna_timeout,
            )
            elapsed = time.time() - t0
            logger.info(f"[Phase 1] 蓝球搜索完成 | 耗时={elapsed:.0f}s ({elapsed/60:.1f}min)")

        # 最终训练
        logger.info("=" * 55)
        logger.info(f"Phase 2/2: 训练最终 Stacking 蓝球模型")
        logger.info("=" * 55)
        t0 = time.time()
        blue_model, blue_metrics = train_blue_model(
            blue_X, blue_y, blue_params, config,
        )
        elapsed = time.time() - t0
        logger.info(f"[Phase 2] 蓝球训练完成 | 耗时={elapsed:.0f}s")
        logger.info(f"[Phase 2] 蓝球测试集: Top-1准确率={blue_metrics['accuracy']:.4f} "
                     f"(随机基线=0.0625, 提升={blue_metrics['accuracy']/0.0625:.1f}倍)")
        logger.info(f"[Phase 2] 蓝球测试集: Top-3准确率={blue_metrics['top3_accuracy']:.4f} "
                     f"(随机基线=0.1875, 提升={blue_metrics['top3_accuracy']/0.1875:.1f}倍)")

        # 保存
        blue_save_path = blue_model.save(
            str(config.model_dir_path / "blue"), version=version,
        )

        # 保存参数
        params_path = Path(blue_save_path) / "params.json"
        params_path.write_text(
            json.dumps({
                "version": version,
                "base_params": blue_params,
                "metrics": blue_metrics,
                "n_features": int(blue_X.shape[1]),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── 汇总 ──
    total_elapsed = time.time() - t0_total
    logger.info("\n" + "=" * 55)
    logger.info(f"训练全部完成！总耗时 {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    logger.info(f"模型版本: {version}")
    logger.info(f"模型目录: {config.model_dir_path}")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
