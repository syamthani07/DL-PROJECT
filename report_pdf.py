"""Visual panel + PDF export for the explainable report."""
import datetime
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ZONE_COLORS = {"upper": (0.30, 0.55, 0.95), "middle": (0.30, 0.80, 0.45), "lower": (0.95, 0.70, 0.25)}


def make_panel(vis, m, path=None):
    """3 panels: X-ray | Grad-CAM overlay | lung zones with attention share."""
    img = vis["img"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))
    for a in ax:
        a.axis("off")
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Chest X-ray")
    ax[1].imshow(img, cmap="gray"); ax[1].imshow(vis["cam"], cmap="jet", alpha=0.40)
    ax[1].set_title(f"Grad-CAM  (p = {m['p_main']:.2f})")
    ax[2].imshow(img, cmap="gray")
    overlay = np.zeros(img.shape + (4,))
    for key, zm in vis["zones"].items():
        overlay[zm] = (*ZONE_COLORS[key.split()[1]], 0.35)
    ax[2].imshow(overlay)
    for key, zm in vis["zones"].items():
        if zm.any():
            ys, xs = np.where(zm)
            ax[2].text(xs.mean(), ys.mean(), f"{m['zone_share'][key]*100:.0f}%", color="white",
                       ha="center", va="center", fontsize=10, weight="bold")
    ax[2].set_title("Lung zones: share of attention")
    ax[2].text(5, 218, "R", color="yellow", fontsize=13, weight="bold")
    ax[2].text(210, 218, "L", color="yellow", fontsize=13, weight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    if path:
        fig.savefig(path, dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf


VERDICT_COLOR = {"PNEUMONIA LIKELY": colors.HexColor("#B42318"),
                 "NO PNEUMONIA DETECTED": colors.HexColor("#067647"),
                 "REFER": colors.HexColor("#B54708")}


def make_pdf(report, m, vis, out_path, patient_id="—", model_name=""):
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Title"], fontSize=17, spaceAfter=4)
    sub = ParagraphStyle("s", parent=ss["Normal"], fontSize=8.5, textColor=colors.grey)
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=10, leading=14)
    lab = ParagraphStyle("l", parent=body, fontName="Helvetica-Bold")
    vstyle = ParagraphStyle("v", parent=ss["Heading2"], textColor=VERDICT_COLOR[report["verdict"]])

    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm,
                            topMargin=1.4*cm, bottomMargin=1.4*cm)
    el = [Paragraph("AI Chest X-Ray Screening Report", h),
          Paragraph(f"Patient / image ID: {patient_id} &nbsp;|&nbsp; Generated: "
                    f"{datetime.datetime.now():%d %b %Y, %H:%M} &nbsp;|&nbsp; Model: {model_name}", sub),
          Spacer(1, 8), Paragraph(report["verdict"], vstyle), Spacer(1, 4)]

    panel = make_panel(vis, m)
    el += [Image(panel, width=17.5*cm, height=6.3*cm), Spacer(1, 8)]

    rows = [[Paragraph(k, lab), Paragraph(v, body)] for k, v in report["sections"].items()]
    t = Table(rows, colWidths=[3.6*cm, 14*cm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.lightgrey),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6)]))
    el += [t, Spacer(1, 10)]

    zs = sorted(m["zone_share"].items(), key=lambda kv: -kv[1])
    zrows = [["Lung zone", "Attention share"]] + [[k.title(), f"{v*100:.1f}%"] for k, v in zs]
    zt = Table(zrows, colWidths=[5*cm, 4*cm])
    zt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F6")),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey)]))
    el += [Paragraph("Measurements behind this report", lab), Spacer(1, 4), zt, Spacer(1, 10)]

    el.append(Paragraph(
        "Report built coarse-to-fine: verdict → section plan → measured slots → template sentences. "
        "Every statement is computed from the measurements above; no free-text generation is used. "
        "Screening aid for research/education only — not a diagnosis.", sub))
    doc.build(el)
    return out_path
