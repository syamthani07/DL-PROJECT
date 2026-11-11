"""
report_engine.py
Anatomy-grounded, explainable pneumonia report generator.

Built on TorchXRayVision (Cohen et al., 2022 - https://github.com/mlmed/torchxrayvision):
  * Main model   : DenseNet-121 trained on RSNA Pneumonia data (optionally + our Kermany-fine-tuned head)
  * Second model : DenseNet-121 trained on 7 public datasets ("all" weights)  -> second opinion
  * Segmentation : PSPNet anatomy model (lungs, heart, ...)                  -> zones

Our contribution (the novelty): turn a Grad-CAM heat map into a measured, structured,
coarse-to-fine report where every sentence is computed from a number - nothing is free-generated.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io
import skimage.transform
import torchxrayvision as xrv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ZONES = ["upper", "middle", "lower"]


# --------------------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------------------
class FineTunedHead(nn.Module):
    """Small head trained on Kermany (pediatric) features. See finetune.py."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1)

    def forward(self, feats):          # feats: [B,1024,7,7] (after relu)
        pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        return torch.sigmoid(self.fc(pooled)).squeeze(1)


class Engine:
    def __init__(self, head_path=None, threshold=None):
        self.main = xrv.models.DenseNet(weights="densenet121-res224-rsna").to(DEVICE).eval()
        self.second = xrv.models.DenseNet(weights="densenet121-res224-all").to(DEVICE).eval()
        self.seg = xrv.baseline_models.chestx_det.PSPNet().to(DEVICE).eval()
        self.head = None
        if head_path:
            self.head = FineTunedHead().to(DEVICE)
            self.head.load_state_dict(torch.load(head_path, map_location=DEVICE))
            self.head.eval()
        # threshold: tuned for recall by finetune.py (saved next to the head), else 0.5
        if threshold is None:
            threshold = 0.5
            if head_path:
                import os
                tj = os.path.join(os.path.dirname(head_path), "threshold.json")
                if os.path.exists(tj):
                    threshold = json.load(open(tj))["threshold"]
        self.threshold = threshold
        self.main_name = ("RSNA DenseNet-121 + Kermany fine-tuned head" if self.head
                          else "RSNA DenseNet-121 (pretrained)")

    # ----------------------------------------------------------------------------------
    # 2. Pre-processing
    # ----------------------------------------------------------------------------------
    @staticmethod
    def load_image(path_or_array):
        img = skimage.io.imread(path_or_array) if isinstance(path_or_array, str) else path_or_array
        if img.ndim == 3:
            img = img[..., :3].mean(2)
        maxval = 65535 if img.max() > 255 else 255
        img = xrv.datasets.normalize(img.astype(np.float32), maxval)   # -> [-1024, 1024]
        img = xrv.datasets.XRayCenterCrop()(img[None, ...])[0]
        return img

    @staticmethod
    def to_tensor(img, size):
        r = skimage.transform.resize(img, (size, size), preserve_range=True, anti_aliasing=True)
        return torch.from_numpy(r.astype(np.float32))[None, None].to(DEVICE)

    # ----------------------------------------------------------------------------------
    # 3. Prediction + Grad-CAM
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _gradcam(model, x, score_fn):
        """Grad-CAM on DenseNet final feature block. Returns (score, cam[224x224] in 0..1)."""
        feats = F.relu(model.features(x))
        feats.retain_grad()
        score = score_fn(feats)
        model.zero_grad()
        score.backward()
        weights = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feats).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return float(score.detach().cpu()), cam

    def _xrv_score_fn(self, model, name):
        idx = model.pathologies.index(name)

        def fn(feats):
            out = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            out = model.classifier(out)
            out = torch.sigmoid(out)
            if hasattr(model, "op_threshs") and model.op_threshs is not None:
                out = xrv.models.op_norm(out, model.op_threshs)   # calibrated: 0.5 = operating point
            return out[0, idx]
        return fn

    def predict_main(self, x):
        if self.head is not None:
            return self._gradcam(self.main, x, lambda f: self.head(f)[0])
        return self._gradcam(self.main, x, self._xrv_score_fn(self.main, "Pneumonia"))

    def predict_second(self, x):
        return self._gradcam(self.second, x, self._xrv_score_fn(self.second, "Pneumonia"))

    @torch.no_grad()
    def other_findings(self, x):
        out = self.second(x)[0].cpu().numpy()
        return {p: float(v) for p, v in zip(self.second.pathologies, out) if p}

    # ----------------------------------------------------------------------------------
    # 4. Anatomy: lungs -> 6 zones
    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def segment(self, img):
        x512 = self.to_tensor(img, 512)
        pred = torch.sigmoid(self.seg(x512))[0].cpu().numpy()       # [14,512,512]
        t = self.seg.targets
        masks = {}
        for name in ["Left Lung", "Right Lung", "Heart"]:
            m = pred[t.index(name)] > 0.5
            masks[name] = skimage.transform.resize(m.astype(float), (224, 224), order=0) > 0.5
        return masks

    @staticmethod
    def split_zones(lung_mask):
        """Split one lung into upper / middle / lower thirds by its own height."""
        rows = np.where(lung_mask.any(1))[0]
        zones = {}
        if len(rows) == 0:
            return {z: np.zeros_like(lung_mask) for z in ZONES}
        top, bottom = rows.min(), rows.max() + 1
        cuts = np.linspace(top, bottom, 4).astype(int)
        for i, z in enumerate(ZONES):
            band = np.zeros_like(lung_mask)
            band[cuts[i]:cuts[i + 1], :] = True
            zones[z] = lung_mask & band
        return zones

    # ----------------------------------------------------------------------------------
    # 5. Measurements (the numbers every sentence is built from)
    # ----------------------------------------------------------------------------------
    def analyse(self, path_or_array):
        img = self.load_image(path_or_array)
        x = self.to_tensor(img, 224)
        img224 = x[0, 0].detach().cpu().numpy()

        p_main, cam = self.predict_main(x)
        p_second, cam2 = self.predict_second(x)
        findings = self.other_findings(x)
        masks = self.segment(img)

        left, right = masks["Left Lung"], masks["Right Lung"]
        lungs = left | right
        total = cam.sum() + 1e-8

        # Orientation check: on a PA film the patient's LEFT lung appears on the IMAGE RIGHT.
        lx = np.where(left)[1].mean() if left.any() else 0
        rx = np.where(right)[1].mean() if right.any() else 0
        anatomical_labels = lx > rx

        zone_masks, zone_share = {}, {}
        for side, m in [("right", right), ("left", left)]:
            for z, zm in self.split_zones(m).items():
                key = f"{side} {z}"
                zone_masks[key] = zm
                zone_share[key] = float((cam * zm).sum() / total)

        hot = cam >= 0.5
        hot2 = cam2 >= 0.5
        inter = (hot & hot2).sum()
        union = (hot | hot2).sum() + 1e-8

        extent = {s: float((hot & m).sum() / (m.sum() + 1e-8)) for s, m in [("right", right), ("left", left)]}

        # Relative density: brightness of highlighted lung vs rest of lung (radiographic opacity)
        hot_lung = hot & lungs
        rest = lungs & ~hot
        density = (float(img224[hot_lung].mean() - img224[rest].mean()) / 1024.0
                   if hot_lung.any() and rest.any() else 0.0)

        m = {
            "p_main": p_main,
            "p_second": p_second,
            "threshold": self.threshold,
            "in_lung_attention": float((cam * lungs).sum() / total),
            "zone_share": zone_share,
            "extent": extent,
            "location_iou": float(inter / union),
            "relative_density": density,
            "lungs_found": bool(left.any() and right.any()),
            "orientation_ok": bool(anatomical_labels),
            "other_findings": findings,
        }
        visuals = {"img": img224, "cam": cam, "cam2": cam2, "zones": zone_masks, "lungs": lungs}
        return m, visuals


# --------------------------------------------------------------------------------------
# 6. Coarse-to-fine report generation (inspired by Next-Concept Modeling, Stanford 2026)
#    Level 1 verdict -> Level 2 section plan -> Level 3 slot filling -> Level 4 sentences
# --------------------------------------------------------------------------------------
def band(p, t=0.5):
    """Confidence = distance from the decision threshold, scaled to the room on that side."""
    d = (p - t) / (1 - t) if p >= t else (t - p) / t
    return "high" if d >= 0.6 else "moderate" if d >= 0.3 else "low"


def level1_verdict(m):
    t = m["threshold"]
    main_pos, second_pos = m["p_main"] >= t, m["p_second"] >= 0.5
    # Reliability gate only matters when the model is calling (or nearly calling) pneumonia:
    # for a confident negative there is no "pneumonia evidence" for Grad-CAM to locate.
    reliable = m["in_lung_attention"] >= 0.5 and m["lungs_found"]
    if not m["lungs_found"]:
        return "REFER", "Lung fields could not be segmented (image quality / positioning)."
    if (main_pos or band(m["p_main"], t) == "low") and not reliable:
        return "REFER", "Model attention is not on the lung fields; result is not reliable."
    if main_pos != second_pos and band(m["p_main"], t) != "high":
        return "REFER", "The two independent models disagree."
    if band(m["p_main"], t) == "low":
        return "REFER", "Prediction is close to the decision threshold."
    return ("PNEUMONIA LIKELY", "") if main_pos else ("NO PNEUMONIA DETECTED", "")


def level2_plan(verdict, m):
    plan = ["result"]
    if verdict != "NO PNEUMONIA DETECTED":
        plan.append("location")
    if verdict != "NO PNEUMONIA DETECTED":
        plan.append("reliability")
    plan.append("agreement")
    if any(v >= 0.5 for k, v in m["other_findings"].items() if k not in ("Pneumonia",)):
        plan.append("other")
    plan.append("recommendation")
    return plan


def level3_slots(verdict, reason, m):
    shares = sorted(m["zone_share"].items(), key=lambda kv: -kv[1])
    top = [(k, v) for k, v in shares if v >= 0.15][:3]
    sides = {k.split()[0] for k, _ in top}
    lat = "bilateral" if len(sides) == 2 else (f"{next(iter(sides))}-sided" if sides else "none")
    others = sorted([(k, v) for k, v in m["other_findings"].items()
                     if v >= 0.5 and k not in ("Pneumonia",)], key=lambda kv: -kv[1])[:4]
    return {
        "verdict": verdict, "reason": reason, "t": m["threshold"],
        "p": m["p_main"], "conf": band(m["p_main"], m["threshold"]),
        "top_zones": top, "laterality": lat,
        "extent_r": m["extent"]["right"], "extent_l": m["extent"]["left"],
        "in_lung": m["in_lung_attention"],
        "p2": m["p_second"], "agree": (m["p_main"] >= m["threshold"]) == (m["p_second"] >= 0.5),
        "iou": m["location_iou"], "density": m["relative_density"],
        "others": others,
    }


def level4_realise(plan, s):
    """Every sentence below is a template filled with a measured value."""
    out = {}
    for sec in plan:
        if sec == "result":
            out["Result"] = (f"{s['verdict']}. Pneumonia probability {s['p']:.2f} "
                             f"(threshold {s['t']:.2f}, {s['conf']} confidence)." + (f" {s['reason']}" if s["reason"] else ""))
        elif sec == "location":
            if s["top_zones"]:
                z = ", ".join(f"{k} zone ({v*100:.0f}%)" for k, v in s["top_zones"])
                out["Location"] = (f"Model attention concentrated in the {z}; pattern is {s['laterality']}. "
                                   f"Highlighted area covers {s['extent_r']*100:.0f}% of the right lung and "
                                   f"{s['extent_l']*100:.0f}% of the left lung.")
            else:
                out["Location"] = "No single lung zone dominates the model's attention (diffuse pattern)."
        elif sec == "reliability":
            ok = s["in_lung"] >= 0.5
            out["Reliability"] = (f"{s['in_lung']*100:.0f}% of the model's attention falls inside the lung fields - "
                                  + ("the prediction is based on lung tissue." if ok else
                                     "the model may be relying on non-lung cues (labels, edges, devices)."))
        elif sec == "agreement":
            out["Second opinion"] = (f"Independent model (trained on 7 other datasets): probability {s['p2']:.2f} - "
                                     f"{'AGREES' if s['agree'] else 'DISAGREES'}. Location overlap "
                                     f"(IoU of highlighted regions) {s['iou']:.2f}.")
        elif sec == "other":
            out["Other signals screened"] = ", ".join(f"{k} ({v:.2f})" for k, v in s["others"]) + \
                ". Scores above 0.5 exceed the model's calibrated operating point."
        elif sec == "recommendation":
            out["Recommendation"] = {
                "PNEUMONIA LIKELY": "Priority review by a clinician.",
                "NO PNEUMONIA DETECTED": "Routine review. Screening aid only - does not rule out disease.",
                "REFER": "Refer for expert radiologist reading.",
            }[s["verdict"]]
    return out


def generate_report(m):
    verdict, reason = level1_verdict(m)
    plan = level2_plan(verdict, m)
    slots = level3_slots(verdict, reason, m)
    text = level4_realise(plan, slots)
    return {"verdict": verdict, "plan": plan, "sections": text}


if __name__ == "__main__":
    import sys
    eng = Engine()
    m, _ = eng.analyse(sys.argv[1])
    r = generate_report(m)
    print(json.dumps({k: v for k, v in m.items() if k != "other_findings"}, indent=2))
    print("\nPLAN:", r["plan"])
    for k, v in r["sections"].items():
        print(f"\n{k.upper()}: {v}")
