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

up = st.file_uploader("Upload a chest X-ray: browse, drag and drop anywhere on the page, or paste a screenshot (Ctrl+V)",
                      type=["png", "jpg", "jpeg", "webp", "bmp"])

# Streamlit's uploader has no clipboard support and only accepts drops on its own box.
# This forwards an image pasted or dropped anywhere on the page into the uploader's file input.
st.html("""
<script>
(() => {
  if (window.__xrayPasteDrop) return;          // register once, not on every rerun
  window.__xrayPasteDrop = true;
  const DROPZONE = '[data-testid="stFileUploaderDropzone"]';
  const images = (list) => Array.from(list || []).filter((f) => f.type.startsWith("image/"));
  const send = (file) => {
    const input = document.querySelector('[data-testid="stFileUploader"] input[type="file"]');
    if (!input) return;
    const ext = (file.type.split("/")[1] || "png").replace("jpeg", "jpg");
    const named = file.name && file.name !== "image.png" ? file
      : new File([file], `screenshot-${new Date().toISOString().slice(11, 19).replaceAll(":", "")}.${ext}`,
                 { type: file.type });
    const dt = new DataTransfer();
    dt.items.add(named);
    input.files = dt.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };
  window.addEventListener("paste", (e) => {
    const [file] = images(e.clipboardData && e.clipboardData.files);
    if (file) { e.preventDefault(); send(file); }
  });
  window.addEventListener("dragover", (e) => {
    if (e.dataTransfer && Array.from(e.dataTransfer.types).includes("Files")) e.preventDefault();
  });
  window.addEventListener("drop", (e) => {
    if (e.target.closest && e.target.closest(DROPZONE)) return;   // the uploader handles its own box
    const [file] = images(e.dataTransfer && e.dataTransfer.files);
    e.preventDefault();                                            // never navigate away to the file
    if (file) send(file);
  });
})();
</script>
""", unsafe_allow_javascript=True)

if not up:
    st.info("Upload, drop or paste an X-ray to generate the report.")
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
    if rep["conditions"]:
        st.subheader("About the flagged findings")
        st.caption("General information about each signal the models flagged (model score in brackets). "
                   "A flagged signal is not a diagnosis.")
        for name, score, text in rep["conditions"]:
            st.markdown(f"**{name}** ({score:.2f}) is {text}")
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
