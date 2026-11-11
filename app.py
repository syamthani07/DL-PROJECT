"""
app.py - demo UI.   Run:  streamlit run app.py
Upload a chest X-ray -> verdict, Grad-CAM, lung-zone map, grounded report, PDF download.
"""
import json
import os
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
from report_engine import Engine, generate_report, level1_verdict, level2_plan, level3_slots
from report_pdf import make_panel, make_pdf

st.set_page_config(page_title="Explainable Pneumonia Screening", page_icon="🫁", layout="wide")

HEAD = "weights/kermany_head.pt"


@st.cache_resource(show_spinner="Loading models (first run downloads weights)...")
def get_engine():
    return Engine(head_path=HEAD if os.path.exists(HEAD) else None)


eng = get_engine()
st.title("🫁 Explainable Pneumonia Screening with Anatomy-Grounded Reports")
st.caption(f"Main model: {eng.main_name} · Decision threshold {eng.threshold:.2f} · "
           "Second opinion: TorchXRayVision 'all' DenseNet · Segmentation: PSPNet anatomy model")

up = st.file_uploader("Upload a chest X-ray (PNG / JPG)", type=["png", "jpg", "jpeg"])
if not up:
    st.info("Upload an X-ray to generate the report.")
    st.stop()

arr = np.array(Image.open(up).convert("L"))
with st.spinner("Analysing: prediction → Grad-CAM → lung segmentation → zone measurements → report"):
    m, vis = eng.analyse(arr)
    rep = generate_report(m)

color = {"PNEUMONIA LIKELY": "#B42318", "NO PNEUMONIA DETECTED": "#067647", "REFER": "#B54708"}[rep["verdict"]]
st.markdown(f"<h2 style='color:{color};margin-bottom:0'>{rep['verdict']}</h2>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pneumonia probability", f"{m['p_main']:.2f}")
c2.metric("Attention inside lungs", f"{m['in_lung_attention']*100:.0f}%")
c3.metric("Second model", f"{m['p_second']:.2f}",
          "agrees" if (m['p_main'] >= m['threshold']) == (m['p_second'] >= 0.5) else "disagrees")
c4.metric("Location overlap (IoU)", f"{m['location_iou']:.2f}")

st.image(make_panel(vis, m), use_container_width=True)

left, right = st.columns([3, 2])
with left:
    st.subheader("Report")
    for k, v in rep["sections"].items():
        st.markdown(f"**{k}** — {v}")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        make_pdf(rep, m, vis, f.name, patient_id=up.name, model_name=eng.main_name)
        st.download_button("⬇️ Download PDF report", open(f.name, "rb").read(),
                           file_name=f"report_{os.path.splitext(up.name)[0]}.pdf", mime="application/pdf")

with right:
    st.subheader("How the report was built (coarse → fine)")
    v, reason = level1_verdict(m)
    plan = level2_plan(v, m)
    slots = level3_slots(v, reason, m)
    st.markdown(f"**Level 1 · Verdict:** `{v}`" + (f" — {reason}" if reason else ""))
    st.markdown(f"**Level 2 · Section plan:** {' → '.join(plan)}")
    st.markdown("**Level 3 · Measured slots:**")
    st.json({k: (round(val, 3) if isinstance(val, float) else val) for k, val in slots.items()
             if k not in ("reason",)}, expanded=False)
    st.markdown("**Level 4 · Sentences:** filled from templates (left). No free-text generation → no hallucination.")
    with st.expander("All raw measurements"):
        st.json(json.loads(json.dumps(m, default=float)))

st.caption("Research / education prototype. Not a medical device. Built on TorchXRayVision (Cohen et al.).")
