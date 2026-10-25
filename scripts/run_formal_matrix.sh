#!/bin/bash
# 正式矩阵驱动(优先级 1,REPORT.md §5.1):
#   阶段 1 控制矩阵:4 范式 × n=100 × max_steps=8(offsets 0-29 已有 n=30,补 30-99)
#   阶段 2 guard-nudge 臂:4 范式 × n=100(react 0-29 已有,补 30-99;其余三范式全量 0-99)
# 幂等可续跑:输出文件已存在的 batch 自动跳过。
set -u
cd "D:/claude code use/agent-research-project" || exit 1
export CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1

DATA=data/raw/hotpotqa/hotpot_dev_distractor_v1.json
BATCH_DIR=experiments/results/matrix/ms100_batches
LOG=experiments/results/matrix/ms100_run.log
mkdir -p "$BATCH_DIR/guard"

run_batch () { # paradigm offset limit guard out trace_dir
  local p=$1 off=$2 lim=$3 guard=$4 out=$5 tr=$6
  if [ -f "$out" ]; then
    echo "[skip] $out 已存在" >> "$LOG"
    return 0
  fi
  echo "[run ] $(date '+%F %T') p=$p off=$off lim=$lim guard=$guard" >> "$LOG"
  python scripts/03_run_agentic_rag.py \
    --data "$DATA" --agent "$p" --offset "$off" --limit "$lim" --max-steps 8 \
    --guard "$guard" --out "$out" --trace-dir "$tr" >> "$LOG" 2>&1
  echo "[done] $(date '+%F %T') $out exit=$?" >> "$LOG"
}

# ---- 阶段 1:控制矩阵(guard=off)----
for p in react plan_execute self_ask ircot; do
  for off in 30 40 50 60 70 80 90; do
    run_batch "$p" "$off" 10 off "$BATCH_DIR/${p}_off${off}.json" "experiments/traces/${p}_ms8"
  done
done

# ---- 阶段 2:guard-nudge 臂 ----
for off in 30 40 50 60 70 80 90; do
  run_batch react "$off" 10 nudge "$BATCH_DIR/guard/react_nudge_off${off}.json" "experiments/traces/react_ms8_guard-nudge"
done
for p in plan_execute self_ask ircot; do
  for off in 0 10 20 30 40 50 60 70 80 90; do
    run_batch "$p" "$off" 10 nudge "$BATCH_DIR/guard/${p}_nudge_off${off}.json" "experiments/traces/${p}_ms8_guard-nudge"
  done
done

echo "ALL_DONE $(date '+%F %T')" >> "$LOG"
