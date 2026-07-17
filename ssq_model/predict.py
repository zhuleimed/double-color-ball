"""模型预测入口。

加载已训练的模型，预测下一期双色球号码。

运行模式:
  python -m ssq_model.predict                # 标准预测
  python -m ssq_model.predict --beam 10      # 指定 Beam Search 宽度
  python -m ssq_model.predict --no-save      # 仅预测，不保存到数据库
  python -m ssq_model.predict --verbose      # 详细输出（含候选号码和置信度）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── 确保项目根目录在 path 中 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from dotenv import load_dotenv
load_dotenv()

from ssq_model.config import get_config
from ssq_model.ensemble import SSQEnsemble
from ssq_sync.engine import DataEngine
from ssq_sync.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="🔴 双色球号码预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m ssq_model.predict                # 标准预测
  python -m ssq_model.predict --beam 10      # Beam Search 宽度=10
  python -m ssq_model.predict --no-save      # 不保存到数据库
  python -m ssq_model.predict --verbose      # 详细输出
        """,
    )
    parser.add_argument("--beam", type=int, default=5,
                        help="Beam Search 宽度 (默认 5)")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存预测结果到数据库")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细预测信息（候选号码、各位置置信度）")
    args = parser.parse_args()

    config = get_config()

    logger.info("=" * 55)
    logger.info(f"双色球预测 | {datetime.now():%Y-%m-%d %H:%M}")
    logger.info("=" * 55)

    # ── 加载模型 ──
    ensemble = SSQEnsemble(config)

    try:
        ensemble.load_models()
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.error("请先运行 train.py 训练模型：python -m ssq_model.train --quick")
        sys.exit(1)

    # ── 加载数据 ──
    engine = DataEngine()
    df = engine.get_all_draws_df()
    logger.info(f"数据库: {len(df)} 期, 最新: {df['period'].iloc[-1]} "
                 f"({df['draw_date'].iloc[-1]})")

    # ── 预测 ──
    ticket = ensemble.predict_next(beam_width=args.beam, df=df)

    # ── 输出 ──
    reds_str = " ".join(f"{r:02d}" for r in ticket["reds"])
    print()
    print("╔══════════════════════════════════╗")
    print("║      🔴 双色球预测号码           ║")
    print("╠══════════════════════════════════╣")
    print(f"║  期号: {ticket['period']:>22s}    ║")
    print(f"║  红球: {reds_str:>22s}    ║")
    print(f"║  蓝球: {ticket['blue']:>22d}    ║")
    print("╚══════════════════════════════════╝")
    print()

    if args.verbose:
        print("📊 红球各位置置信度:")
        for i, (r, c) in enumerate(zip(ticket["reds"], ticket["red_confidence"])):
            bar = "█" * int(c * 20) + "░" * (20 - int(c * 20))
            print(f"  位置{i+1}: {r:02d}  [{bar}] {c:.2%}")

        print(f"\n🔵 蓝球置信度: {ticket['blue_confidence']:.2%}")
        if ticket.get("blue_top3"):
            print(f"  蓝球 Top-3: {' '.join(f'{b:02d}' for b in ticket['blue_top3'])}")

        if ticket.get("red_candidates") and len(ticket["red_candidates"]) > 1:
            print(f"\n🔍 备选红球组合:")
            for i, cand in enumerate(ticket["red_candidates"][1:], 1):
                print(f"  候选{i}: {' '.join(f'{r:02d}' for r in cand)}")

    # ── 保存到数据库 ──
    if not args.no_save:
        row_id = ensemble.save_prediction(ticket)
        logger.info(f"预测已保存到数据库, ID={row_id}")
    else:
        logger.info("跳过保存（--no-save）")

    # ── 输出 JSON（供 pipeline 程序化使用） ──
    output = {
        "period": ticket["period"],
        "pred_date": datetime.now().strftime("%Y-%m-%d"),
        "reds": ticket["reds"],
        "blue": ticket["blue"],
        "red_confidence": ticket["red_confidence"],
        "blue_confidence": ticket["blue_confidence"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
