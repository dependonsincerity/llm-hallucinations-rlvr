import argparse
import pandas as pd
import yaml
import re

def normalize(s):
    if s is None:
        s = ""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    return s

def token_f1(pred, gold):
    pred_toks = set(normalize(pred).split())
    gold_toks = set(normalize(gold).split())
    if not pred_toks or not gold_toks:
        return 0.0
    inter = len(pred_toks & gold_toks)
    prec = inter / len(pred_toks)
    rec = inter / len(gold_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def evaluate(df, threshold):
    scores = []
    for _, row in df.iterrows():
        golds = str(row["gold"]).split("|")
        f1s = [token_f1(row["answer"], g) for g in golds]
        best = max(f1s)
        scores.append(best)
    acc = sum(1 for s in scores if s >= threshold) / len(scores) if scores else 0.0
    return acc, scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="e.g. qwen_baseline_outputs")
    args = parser.parse_args()

    with open("configs/eval.yaml", "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    path = f"results/logs/{args.run_id}.csv"
    df = pd.read_csv(path)

    res = []
    for ds_name, ds_df in df.groupby("dataset"):
        acc, scores = evaluate(ds_df, eval_cfg["evaluation"]["token_f1_threshold"])
        res.append({"dataset": ds_name, "accuracy@f1": acc, "n": len(ds_df)})

    out_path = f"results/tables/{args.run_id}_metrics.csv"
    pd.DataFrame(res).to_csv(out_path, index=False)
    print("Saved:", out_path)
