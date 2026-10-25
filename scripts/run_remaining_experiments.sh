#!/bin/bash
# 项目收尾实验总驱动(幂等可续跑,已存在的 batch 自动跳过):
#   阶段 1  react 消融:ms=4 + 全部修复(截断修复贡献拆分,§5.4)
#   阶段 2  self_ask 扩样 n=500(offsets 100-499,0-99 已有)
#   阶段 3  2WikiMultihopQA 泛化:数据准备 → 索引 → baseline → 4 范式 × n=100 × ms=8
#   阶段 4  MuSiQue 泛化:同上
set -u
cd "D:/claude code use/agent-research-project" || exit 1
export CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1

HOTPOT=data/raw/hotpotqa/hotpot_dev_distractor_v1.json
LOG=experiments/results/matrix/wrapup_run.log
echo "=== WRAPUP START $(date '+%F %T') ===" >> "$LOG"

run () { # 标签 + 完整命令
  local tag=$1; shift
  echo "[run ] $(date '+%F %T') $tag" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "[done] $(date '+%F %T') $tag exit=$?" >> "$LOG"
}

# ---- 阶段 1:react ms=4 消融(全部修复在位,与 v2/ms4 差异 = 观察截断修复)----
mkdir -p experiments/results/matrix/ms4fixed_batches
for off in 0 10 20; do
  out=experiments/results/matrix/ms4fixed_batches/react_off${off}.json
  [ -f "$out" ] && { echo "[skip] $out" >> "$LOG"; continue; }
  run "react_ms4 off=$off" python scripts/03_run_agentic_rag.py \
    --data "$HOTPOT" --agent react --offset $off --limit 10 --max-steps 4 \
    --out "$out" --trace-dir experiments/traces/react_ms4_fixed
done

# ---- 阶段 2:self_ask n=500(补 offsets 100-499)----
mkdir -p experiments/results/matrix/n500_batches
for off in $(seq 100 10 490); do
  out=experiments/results/matrix/n500_batches/self_ask_off${off}.json
  [ -f "$out" ] && { echo "[skip] $out" >> "$LOG"; continue; }
  run "self_ask_n500 off=$off" python scripts/03_run_agentic_rag.py \
    --data "$HOTPOT" --agent self_ask --offset $off --limit 10 --max-steps 8 \
    --out "$out" --trace-dir experiments/traces/self_ask_ms8
done

# ---- 阶段 3/4:泛化数据集 ----
for DS in 2wiki musique; do
  case $DS in
    2wiki)   CONV=data/processed/2wiki_dev_hotpot_schema.json;  IDX=data/processed/index/2wiki_faiss;  BASE=experiments/results/baseline_2wiki.json;  BDIR=experiments/results/matrix/2wiki_batches;  TRSUF=_2wiki ;;
    musique) CONV=data/processed/musique_dev_hotpot_schema.json; IDX=data/processed/index/musique_faiss; BASE=experiments/results/baseline_musique.json; BDIR=experiments/results/matrix/musique_batches; TRSUF=_musique ;;
  esac

  run "$DS prepare" python scripts/14_prepare_dataset.py --dataset $DS
  if [ ! -f "${IDX}.docs.npy" ]; then
    run "$DS build_index" python scripts/01_build_index.py --data "$CONV" --out "$IDX"
  else
    echo "[skip] $IDX 已存在" >> "$LOG"
  fi
  if [ ! -f "$BASE" ]; then
    run "$DS baseline" python scripts/02_run_baseline_rag.py \
      --data "$CONV" --index "$IDX" --out "$BASE" --limit 100
  else
    echo "[skip] $BASE 已存在" >> "$LOG"
  fi
  mkdir -p "$BDIR"
  for p in react plan_execute self_ask ircot; do
    for off in 0 10 20 30 40 50 60 70 80 90; do
      out="$BDIR/${p}_off${off}.json"
      [ -f "$out" ] && { echo "[skip] $out" >> "$LOG"; continue; }
      run "$DS $p off=$off" python scripts/03_run_agentic_rag.py \
        --data "$CONV" --index "$IDX" --agent $p --offset $off --limit 10 --max-steps 8 \
        --out "$out" --trace-dir "experiments/traces/${p}${TRSUF}"
    done
  done
done

echo "ALL_DONE $(date '+%F %T')" >> "$LOG"
