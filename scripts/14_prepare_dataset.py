"""准备 2WikiMultihopQA / MuSiQue 泛化实验数据:下载(如缺)+ 转成 HotpotQA schema。

转换后输出 data/processed/{ds}_dev_hotpot_schema.json,使
01_build_index / 02_run_baseline_rag / 03_run_agentic_rag / 08_replay_gold_hit
等下游脚本零改动复用(load_hotpotqa 可直接加载)。

- 2wiki: supporting_facts 由 [title, sent_idx] 转为 {title, sent_id};
- musique: supporting titles 取 is_supporting 段落;每段 paragraph_text 作为
  单"句"文档(段落级检索,max_doc_chars=2000 会截断超长段);level = 跳数。

用法:python scripts/14_prepare_dataset.py --dataset {2wiki,musique,all}
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "2wiki": (
        "data/raw/2wikimultihop/dev.json",
        "https://hf-mirror.com/datasets/voidful/2WikiMultihopQA/resolve/main/dev.json",
    ),
    "musique": (
        "data/raw/musique/musique_ans_v1.0_dev.jsonl",
        "https://hf-mirror.com/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl",
    ),
}


def download_if_missing(raw_path: Path, url: str) -> None:
    if raw_path.exists():
        print(f"[skip] {raw_path} 已存在")
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[down] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, raw_path.open("wb") as f:
        f.write(r.read())
    print(f"[ok] {raw_path} ({raw_path.stat().st_size / 1e6:.1f} MB)")


def convert_2wiki() -> list[dict]:
    raw = json.load(open(SOURCES["2wiki"][0], encoding="utf-8"))
    out = []
    for q in raw:
        out.append({
            "_id": q["_id"],
            "question": q["question"],
            "answer": q.get("answer", ""),
            "supporting_facts": [
                {"title": t, "sent_id": s} for t, s in q.get("supporting_facts", [])
            ],
            "context": q.get("context", []),
            "type": q.get("type", ""),
            "level": "",
        })
    return out


def convert_musique() -> list[dict]:
    out = []
    with open(SOURCES["musique"][0], encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            out.append({
                "_id": m.get("id") or m.get("question_id", ""),
                "question": m["question"],
                "answer": m.get("answer", ""),
                "supporting_facts": [
                    {"title": p["title"], "sent_id": 0}
                    for p in m["paragraphs"] if p.get("is_supporting")
                ],
                # 每个段落整体作为一个"句"级文档(段落级检索)
                "context": [[p["title"], [p["paragraph_text"]]] for p in m["paragraphs"]],
                "type": "musique",
                "level": f"{len(m.get('question_decomposition', []))}-hop",
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="准备泛化数据集(HotpotQA schema)")
    parser.add_argument("--dataset", choices=["2wiki", "musique", "all"], default="all")
    args = parser.parse_args()

    names = ["2wiki", "musique"] if args.dataset == "all" else [args.dataset]
    for name in names:
        raw_path, url = SOURCES[name]
        download_if_missing(Path(ROOT / raw_path), url)
        out_path = ROOT / "data" / "processed" / f"{name}_dev_hotpot_schema.json"
        if out_path.exists():
            n = len(json.load(open(out_path, encoding="utf-8")))
            print(f"[skip] {out_path} 已存在(n={n})")
            continue
        recs = convert_2wiki() if name == "2wiki" else convert_musique()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False)
        n_gold = sum(1 for r in recs if r["supporting_facts"])
        print(f"[ok] {out_path}: n={len(recs)}, 有金标支撑 {n_gold}")


if __name__ == "__main__":
    main()
