"""
evaluate.py - numbers for your slides, on the Kermany TEST set (624 independent patients).

Compares:  (1) pretrained RSNA model as-is (adult-trained)   vs   (2) + our Kermany fine-tuned head
and measures the report pipeline itself (REFER rate, in-lung attention, model agreement).

Usage: python evaluate.py --data /path/to/chest_xray [--head weights/kermany_head.pt]
Outputs: results/metrics.json, results/confusion.png, results/report_stats.json
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from report_engine import Engine, generate_report
from finetune import list_images


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": round(float(t), 3), "accuracy": round((tp + tn) / len(y), 4),
            "recall": round(tp / (tp + fn), 4), "specificity": round(tn / (tn + fp), 4),
            "precision": round(tp / (tp + fp + 1e-9), 4), "auc": round(roc_auc_score(y, p), 4),
            "missed_pneumonia": int(fn), "false_alarms": int(fp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--head", default=None)
    ap.add_argument("--limit", type=int, default=0, help="quick test on N images")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    items = list_images(os.path.join(a.data, "test"))
    if a.limit:
        items = items[:a.limit // 2] + items[-a.limit // 2:]

    base = Engine()
    tuned = Engine(head_path=a.head) if a.head else None
    eng = tuned or base
    y, p_base, p_tuned, verdicts, inlung, agree = [], [], [], [], [], []
    for i, (path, label) in enumerate(items):
        x = base.to_tensor(base.load_image(path), 224)
        p_base.append(base.predict_main(x)[0])
        m, _ = eng.analyse(path)
        if tuned:
            p_tuned.append(m["p_main"])
        r = generate_report(m)
        y.append(label); verdicts.append(r["verdict"]); inlung.append(m["in_lung_attention"])
        agree.append((m["p_main"] >= m["threshold"]) == (m["p_second"] >= 0.5))
        print(f"\r{i+1}/{len(items)}", end="")
    print()
    y = np.array(y)

    res = {"n_test": len(y), "pretrained_rsna": metrics(y, np.array(p_base), 0.5)}
    if tuned:
        res["finetuned_kermany"] = metrics(y, np.array(p_tuned), tuned.threshold)

    # Report-level statistics: does "REFER" catch the hard cases?
    v = np.array(verdicts)
    called = v != "REFER"
    pred_called = (v[called] == "PNEUMONIA LIKELY").astype(int)
    stats = {
        "refer_rate": round(float((v == "REFER").mean()), 4),
        "accuracy_on_non_referred": round(float((pred_called == y[called]).mean()), 4) if called.any() else None,
        "recall_on_non_referred": round(float(pred_called[y[called] == 1].mean()), 4) if (y[called] == 1).any() else None,
        "model_agreement_rate": round(float(np.mean(agree)), 4),
        "mean_in_lung_attention_pneumonia": round(float(np.mean(np.array(inlung)[y == 1])), 4),
    }
    res["report_pipeline"] = stats
    json.dump(res, open("results/metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    # Confusion matrices
    cols = [("Pretrained (adult RSNA)", np.array(p_base), 0.5)]
    if tuned:
        cols.append(("Fine-tuned on Kermany", np.array(p_tuned), tuned.threshold))
    fig, axs = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4.2))
    axs = np.atleast_1d(axs)
    for ax, (name, p, t) in zip(axs, cols):
        cm = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (r, c), val in np.ndenumerate(cm):
            ax.text(c, r, val, ha="center", va="center", fontsize=14,
                    color="white" if val > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Normal", "Pneumonia"]); ax.set_yticks([0, 1], ["Normal", "Pneumonia"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
    plt.tight_layout(); plt.savefig("results/confusion.png", dpi=140)
    print("Saved results/metrics.json and results/confusion.png")


if __name__ == "__main__":
    main()
