"""
finetune.py - light fine-tuning of the RSNA DenseNet on the Kermany pediatric dataset.

Why: TorchXRayVision's RSNA model was trained on ADULT X-rays; Kermany is PEDIATRIC.
We freeze the backbone (keeps what it learned about chest X-rays) and train a new
1024 -> 1 head on Kermany features. Then we pick the decision threshold on a held-out
split so that recall >= TARGET_RECALL (missing a sick child is worse than a false alarm).

Usage:  python finetune.py --data /path/to/chest_xray --out weights/
        (folder must contain train/NORMAL, train/PNEUMONIA)
Runtime: ~5-8 min on a Colab T4 GPU.
"""
import argparse, glob, json, os, random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from report_engine import Engine, FineTunedHead, DEVICE
import torch.nn.functional as F

TARGET_RECALL = 0.95


def list_images(root):
    items = []
    for label, cls in [(0, "NORMAL"), (1, "PNEUMONIA")]:
        for ext in ("jpeg", "jpg", "png"):
            items += [(p, label) for p in glob.glob(os.path.join(root, cls, f"*.{ext}"))]
    return items


@torch.no_grad()
def extract(engine, items, bs=32):
    feats, labels = [], []
    batch, blab = [], []
    for i, (p, y) in enumerate(items):
        batch.append(engine.to_tensor(engine.load_image(p), 224)); blab.append(y)
        if len(batch) == bs or i == len(items) - 1:
            x = torch.cat(batch)
            f = F.relu(engine.main.features(x))
            feats.append(F.adaptive_avg_pool2d(f, 1).flatten(1).cpu()); labels += blab
            batch, blab = [], []
            print(f"\r  features {i+1}/{len(items)}", end="")
    print()
    return torch.cat(feats), torch.tensor(labels, dtype=torch.float32)


def pick_threshold(probs, y, target=TARGET_RECALL):
    """Highest threshold whose recall is still >= target."""
    pos = np.sort(probs[y == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k]) if len(pos) else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(42); torch.manual_seed(42); np.random.seed(42)

    engine = Engine()
    items = list_images(os.path.join(a.data, "train"))
    random.shuffle(items)
    n_val = int(0.15 * len(items))
    val_items, tr_items = items[:n_val], items[n_val:]
    print(f"train {len(tr_items)}  val {len(val_items)}")

    Xtr, ytr = extract(engine, tr_items)
    Xva, yva = extract(engine, val_items)

    head = FineTunedHead().to(DEVICE)
    pos_w = torch.tensor([(ytr == 0).sum() / (ytr == 1).sum()]).to(DEVICE)   # class imbalance
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xtr, ytr, Xva_d = Xtr.to(DEVICE), ytr.to(DEVICE), Xva.to(DEVICE)
    best, best_state = 0, None
    for ep in range(a.epochs):
        head.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            loss = lossf(head.fc(Xtr[idx]).squeeze(1), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            pv = torch.sigmoid(head.fc(Xva_d)).squeeze(1).cpu().numpy()
        auc = roc_auc_score(yva.numpy(), pv)
        if auc > best:
            best, best_state = auc, {k: v.clone() for k, v in head.state_dict().items()}
        print(f"epoch {ep+1:2d}  loss {loss.item():.4f}  val AUC {auc:.4f}")

    head.load_state_dict(best_state)
    with torch.no_grad():
        pv = torch.sigmoid(head.fc(Xva_d)).squeeze(1).cpu().numpy()
    t = pick_threshold(pv, yva.numpy())
    torch.save(head.state_dict(), os.path.join(a.out, "kermany_head.pt"))
    json.dump({"threshold": t, "target_recall": TARGET_RECALL, "val_auc": best},
              open(os.path.join(a.out, "threshold.json"), "w"), indent=2)
    print(f"\nSaved head. Best val AUC {best:.4f}. Recall-tuned threshold {t:.3f}")


if __name__ == "__main__":
    main()
