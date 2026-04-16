"""
ITAS — Intelligent Threat Assessment System
Streamlit App | Weapon Detection · Behavior Analysis · Biometric Verification
Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
//import json
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import tempfile, os, time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ITAS — Intelligent Threat Assessment",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
.stApp { background: #0a0c10; color: #c8d6e5; }
h1, h2, h3 { font-family: 'Share Tech Mono', monospace; color: #00e5ff; letter-spacing: 2px; }
.metric-card { background: #0d1117; border: 1px solid #00e5ff33; border-left: 3px solid #00e5ff;
    padding: 14px 18px; border-radius: 4px; margin: 6px 0; }
.metric-card .label { color: #607d8b; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
.metric-card .value { color: #00e5ff; font-size: 26px; font-family: 'Share Tech Mono', monospace; font-weight: bold; }
.alert-danger  { background:#1a0a0a; border-left:3px solid #ff1744; padding:12px 16px; border-radius:4px; color:#ff5252; }
.alert-warning { background:#1a1400; border-left:3px solid #ffab00; padding:12px 16px; border-radius:4px; color:#ffd740; }
.alert-safe    { background:#0a1a0a; border-left:3px solid #00e676; padding:12px 16px; border-radius:4px; color:#69f0ae; }
.module-header { background: linear-gradient(90deg, #00e5ff11, transparent); border-left: 4px solid #00e5ff;
    padding: 10px 16px; margin-bottom: 20px; font-family: 'Share Tech Mono', monospace;
    font-size: 18px; color: #00e5ff; letter-spacing: 2px; }
.stButton>button { background: transparent; border: 1px solid #00e5ff; color: #00e5ff;
    font-family: 'Share Tech Mono', monospace; letter-spacing: 1px; padding: 8px 24px; transition: all 0.2s; }
.stButton>button:hover { background: #00e5ff22; border-color: #00e5ff; color: #fff; }
.sidebar-logo { text-align:center; padding: 20px 0 10px 0; font-family: 'Share Tech Mono', monospace;
    font-size: 22px; color: #00e5ff; letter-spacing: 3px; }
div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #00e5ff22; }
.stTabs [data-baseweb="tab"] { color: #607d8b; font-family: 'Share Tech Mono', monospace; }
.stTabs [aria-selected="true"] { color: #00e5ff !important; border-bottom: 2px solid #00e5ff !important; }
hr { border-color: #00e5ff22; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
MODEL_DIR    = Path("models")
METRICS_PATH = Path("ALL_METRICS.json")

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}

METRICS = load_metrics()

# ─────────────────────────────────────────────
# DARKVISION
# ─────────────────────────────────────────────
def compute_mpi(img_bgr):
    return float(np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)))

def darkvision(img_bgr, thr=55):
    m = compute_mpi(img_bgr)
    if m <= thr:
        gamma = float(np.clip(110.0 / (m + 1e-6), 1.5, 2.5))
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(img_bgr, lut)
        out = cv2.GaussianBlur(out, (5, 5), 1.0)
        return out, True, m, gamma
    return img_bgr, False, m, None

# ─────────────────────────────────────────────
# MODEL DEFINITIONS — EXACT match to notebook
# ─────────────────────────────────────────────

# ── Behavior: VGG16 + LSTM (exact from notebook Cell 20) ──
class BehaviorModel(nn.Module):
    def __init__(self, n_cls=2, feat=512, hid=256, nlayers=2, drop=0.5):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        bb = vgg16(weights=None)
        self.features = bb.features
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, feat),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.lstm = nn.LSTM(feat, hid, nlayers, batch_first=True,
                            dropout=drop if nlayers > 1 else 0.)
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(hid, 128),
            nn.ReLU(),
            nn.Linear(128, n_cls)
        )

    def forward(self, x):          # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        f = self.pool(self.features(x.view(B * T, C, H, W)))
        return self.head(self.lstm(self.proj(f).view(B, T, -1))[0][:, -1, :])


# ── Biometric: FN13 (exact from notebook Cell 24) ──
class FN13(nn.Module):
    def __init__(self, n, emb=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, emb),
            nn.BatchNorm1d(emb)
        )
        self.classifier = nn.Linear(emb, n)

    def forward(self, x):
        e = self.embed(self.features(x))
        return self.classifier(e), e

    def get_embedding(self, x):
        return self.embed(self.features(x))


# ─────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_weapon_model():
    pt = MODEL_DIR / "yolo_weapon.pt"
    if not pt.exists():
        return None
    try:
        import torch
        # PyTorch 2.6 changed weights_only default to True which breaks YOLO full-object saves.
        # Patch load to use weights_only=False before YOLO initialises.
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
        from ultralytics import YOLO
        model = YOLO(str(pt))
        torch.load = _orig_load  # restore original
        return model
    except Exception as e:
        st.warning(f"Weapon model load error: {e}")
        return None

@st.cache_resource
def load_behavior_model():
    pt = MODEL_DIR / "behavior_model.pth"
    if not pt.exists():
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BehaviorModel()
        model.load_state_dict(torch.load(str(pt), map_location=device, weights_only=False))
        model.eval()
        return model.to(device)
    except Exception as e:
        st.warning(f"Behavior model load error: {e}")
        return None

@st.cache_resource
def load_face_model():
    pt = MODEL_DIR / "fn13_face.pth"
    if not pt.exists():
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FN13(n=100, emb=128)
        model.load_state_dict(torch.load(str(pt), map_location=device, weights_only=False))
        model.eval()
        return model.to(device)
    except Exception as e:
        st.warning(f"Face model load error: {e}")
        return None

# ─────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────
BEHAVIORS = ["fighting", "walking"]

def run_weapon_detection(img_bgr, model, conf_thresh=0.25):
    enhanced, was_enhanced, mpi, gamma = darkvision(img_bgr)
    results = model(enhanced, verbose=False, conf=conf_thresh)
    r = results[0]
    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        xyxy   = box.xyxy[0].cpu().numpy().astype(int)
        detections.append({"class": cls_id, "conf": conf, "box": xyxy})
    out = enhanced.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 229, 255), 2)
        cv2.putText(out, f"weapon {d['conf']:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 2)
    return out, detections, was_enhanced, mpi

def run_behavior_analysis(frames_bgr, model):
    from torchvision import transforms
    device = next(model.parameters()).device
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    seq = []
    for f in frames_bgr:
        t = transforms.ToTensor()(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        t = norm(t)
        seq.append(t)
    x = torch.stack(seq).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    return BEHAVIORS[pred], float(probs[pred]), {b: float(p) for b, p in zip(BEHAVIORS, probs)}

def get_face_embedding(img_gray, model):
    device = next(model.parameters()).device
    face = cv2.resize(img_gray, (64, 64)).astype(np.float32) / 255.0
    t = torch.from_numpy(face).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.get_embedding(t)
    return emb[0].cpu().numpy()

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def preprocess_face_image(uploaded_file, use_mtcnn=True):
    arr = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if use_mtcnn:
        try:
            from facenet_pytorch import MTCNN
            mt  = MTCNN(image_size=64, margin=10, keep_all=False, post_process=False)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            face = mt(pil)
            if face is not None:
                face_np = face.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
                gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
                return gray, img
        except Exception:
            pass
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    return gray, img

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown('<div class="sidebar-logo">⬡ ITAS</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")
module = st.sidebar.radio(
    "SELECT MODULE",
    ["📊 Dashboard", "🚨 Unified Threat", "🔫 Weapon Detection", "🎬 Behavior Analysis", "👤 Biometric Verification"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**SYSTEM STATUS**")
for label, path in [("YOLO Weapon",      MODEL_DIR / "yolo_weapon.pt"),
                     ("Behavior VGG+LSTM", MODEL_DIR / "behavior_model.pth"),
                     ("FN13 Face",         MODEL_DIR / "fn13_face.pth")]:
    ok    = path.exists()
    color = "#00e676" if ok else "#ff5252"
    sym   = "●" if ok else "○"
    st.sidebar.markdown(f'<span style="color:{color}">{sym}</span> `{label}`', unsafe_allow_html=True)

device_label = ("CUDA · " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"
st.sidebar.markdown(f'<span style="color:#ffd740">◈</span> `{device_label}`', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# ── DASHBOARD ──
# ═══════════════════════════════════════════════
if module == "📊 Dashboard":
    st.markdown('<h1>◈ ITAS DASHBOARD</h1>', unsafe_allow_html=True)
    if not METRICS:
        st.warning("ALL_METRICS.json not found.")
    else:
        st.markdown(f"**Experiment:** `{METRICS.get('experiment_date','?')}` &nbsp;|&nbsp; **GPU:** `{METRICS.get('gpu','?')}`", unsafe_allow_html=True)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="module-header">🔫 WEAPON DETECTION</div>', unsafe_allow_html=True)
            wd = METRICS.get("weapon_detection", {}).get("summary", {})
            for k, label in [("map50","mAP@50"),("map50_95","mAP@50:95"),("precision","Precision"),("recall","Recall")]:
                v = wd.get(k, {})
                st.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                            f'<div class="value">{v.get("mean",0):.4f} <span style="font-size:14px;color:#546e7a">± {v.get("std",0):.4f}</span></div></div>',
                            unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="module-header">🎬 BEHAVIOR ANALYSIS</div>', unsafe_allow_html=True)
            ba = METRICS.get("behavior_analysis", {}).get("summary", {})
            for k, label in [("accuracy","Accuracy (%)"),("f1","F1 Score"),("specificity","Specificity (%)")]:
                v = ba.get(k, {})
                st.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                            f'<div class="value">{v.get("mean",0):.2f} <span style="font-size:14px;color:#546e7a">± {v.get("std",0):.2f}</span></div></div>',
                            unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="module-header">👤 BIOMETRIC (FN13)</div>', unsafe_allow_html=True)
            bio = METRICS.get("biometric_fn13", {}).get("summary", {})
            for k, label in [("verification_accuracy","Verif. Accuracy (%)"),("roc_auc","ROC AUC"),
                              ("TAR_at_EER_pct","TAR @ EER (%)"),("FAR_at_EER_pct","FAR @ EER (%)")]:
                v = bio.get(k, {})
                st.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                            f'<div class="value">{v.get("mean",0):.3f} <span style="font-size:14px;color:#546e7a">± {v.get("std",0):.3f}</span></div></div>',
                            unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="module-header">⚗ DARKVISION ABLATION</div>', unsafe_allow_html=True)
        ab = METRICS.get("ablation", {})
        dv = METRICS.get("darkvision", {})
        cols = st.columns(len(ab.get("per_level", [])) + 1)
        for i, lv in enumerate(ab.get("per_level", [])):
            imp   = lv["improvement_pp"]
            color = "#00e676" if imp > 0 else "#ff5252"
            with cols[i]:
                st.markdown(f'<div class="metric-card"><div class="label">{lv["level"]} (×{lv["factor"]})</div>'
                            f'<div class="value" style="color:{color}">{imp:+.0f} pp</div>'
                            f'<div style="font-size:12px;color:#607d8b">Base {lv["baseline_det_pct"]}% → DV {lv["darkvision_det_pct"]}%</div></div>',
                            unsafe_allow_html=True)
        with cols[-1]:
            st.markdown(f'<div class="metric-card"><div class="label">DarkVision Enhanced</div>'
                        f'<div class="value">{dv.get("pct",0):.1f}%</div>'
                        f'<div style="font-size:12px;color:#607d8b">{dv.get("enhanced",0)}/{dv.get("total",0)} frames · MPI {dv.get("mean_mpi",0)}</div></div>',
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# ── WEAPON DETECTION ──
# ═══════════════════════════════════════════════
elif module == "🔫 Weapon Detection":
    st.markdown('<h1>🔫 WEAPON DETECTION</h1>', unsafe_allow_html=True)
    model = load_weapon_model()
    if model is None:
        st.markdown('<div class="alert-warning">⚠ Place <code>models/yolo_weapon.pt</code> in the models/ folder.</div>', unsafe_allow_html=True)
        st.stop()

    tab_img, tab_live = st.tabs(["🖼 Upload Image", "📷 Live Webcam"])

    with tab_img:
        conf_thresh = st.slider("Confidence Threshold", 0.01, 0.95, 0.25, 0.01)
        uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**INPUT**")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with st.spinner("Running detection..."):
                t0 = time.time()
                out_img, detections, was_enhanced, mpi = run_weapon_detection(img_bgr, model, conf_thresh)
                elapsed = time.time() - t0
            with col2:
                st.markdown("**DETECTION OUTPUT**")
                st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Weapons Found", len(detections))
            m2.metric("Inference Time", f"{elapsed:.2f}s")
            m3.metric("Mean Pixel Intensity", f"{mpi:.1f}")
            m4.metric("DarkVision Applied", "YES ✓" if was_enhanced else "NO")
            if detections:
                st.markdown('<div class="alert-danger">🚨 WEAPON DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-safe">✓ No weapons detected</div>', unsafe_allow_html=True)

    with tab_live:
        st.markdown("**Live Webcam Detection**")
        st.info("Click **Start** to open your laptop camera. Press **Stop** to end.")
        conf_live = st.slider("Live Confidence Threshold", 0.01, 0.95, 0.25, 0.01, key="live_conf")

        run_live = st.checkbox("▶ Start Live Detection", key="live_weapon")
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        if run_live:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam. Make sure it is connected and not in use.")
            else:
                while st.session_state.get("live_weapon", False):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out_img, detections, was_enhanced, mpi = run_weapon_detection(frame, model, conf_live)
                    # Overlay
                    label_txt = f"THREAT DETECTED ({len(detections)})" if detections else "CLEAR"
                    color     = (0, 0, 255) if detections else (0, 230, 118)
                    cv2.putText(out_img, label_txt, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(out_img, f"MPI:{mpi:.0f} DV:{'ON' if was_enhanced else 'OFF'}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 1)
                    frame_placeholder.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB),
                                            channels="RGB", use_container_width=True)
                    if detections:
                        status_placeholder.markdown('<div class="alert-danger">🚨 WEAPON DETECTED</div>', unsafe_allow_html=True)
                    else:
                        status_placeholder.markdown('<div class="alert-safe">✓ CLEAR</div>', unsafe_allow_html=True)
                cap.release()

# ═══════════════════════════════════════════════
# ── BEHAVIOR ANALYSIS ──
# ═══════════════════════════════════════════════
elif module == "🎬 Behavior Analysis":
    st.markdown('<h1>🎬 BEHAVIOR ANALYSIS</h1>', unsafe_allow_html=True)
    st.markdown("VGG16 + LSTM — Violence / Normal Classification", unsafe_allow_html=True)

    model = load_behavior_model()
    if model is None:
        st.markdown('<div class="alert-warning">⚠ Place <code>models/behavior_model.pth</code> in the models/ folder.</div>', unsafe_allow_html=True)
        st.stop()

    SEQ_LEN = 10
    tab_vid, tab_frames, tab_live = st.tabs(["📹 Upload Video", "🖼 Upload 10 Frames", "📷 Live Webcam"])

    with tab_vid:
        video_file = st.file_uploader("Upload video clip (.mp4, .avi, .mov)", type=["mp4","avi","mov"])
        if video_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name
            st.video(tmp_path)
            if st.button("▶ Analyse Video"):
                cap   = cv2.VideoCapture(tmp_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total < SEQ_LEN:
                    st.error(f"Video too short ({total} frames). Need ≥ {SEQ_LEN}.")
                else:
                    idxs   = np.linspace(0, total - 1, SEQ_LEN, dtype=int)
                    frames = []
                    for idx in idxs:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                        ok, f = cap.read()
                        if ok:
                            frames.append(cv2.resize(f, (224, 224)))
                    cap.release()
                    os.unlink(tmp_path)
                    if len(frames) == SEQ_LEN:
                        with st.spinner("Analysing..."):
                            label, conf, probs = run_behavior_analysis(frames, model)
                        if label == "fighting":
                            st.markdown(f'<div class="alert-danger">🚨 VIOLENT BEHAVIOUR — {conf*100:.1f}%</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="alert-safe">✓ NORMAL BEHAVIOUR — {conf*100:.1f}%</div>', unsafe_allow_html=True)
                        for b, p in probs.items():
                            st.write(f"`{b}`: {p*100:.1f}%")
                            st.progress(float(p))
                        st.markdown("**Sampled Frames:**")
                        cols = st.columns(SEQ_LEN)
                        for c, f in zip(cols, frames):
                            c.image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), use_container_width=True)

    with tab_frames:
        imgs = st.file_uploader("Upload exactly 10 frames", type=["jpg","jpeg","png"],
                                accept_multiple_files=True, key="frames10")
        if imgs and len(imgs) == SEQ_LEN:
            frames = []
            for img_f in imgs:
                arr = np.frombuffer(img_f.read(), np.uint8)
                f   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                frames.append(cv2.resize(f, (224, 224)))
            if st.button("▶ Analyse Frames"):
                with st.spinner("Analysing..."):
                    label, conf, probs = run_behavior_analysis(frames, model)
                if label == "fighting":
                    st.markdown(f'<div class="alert-danger">🚨 VIOLENT — {conf*100:.1f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-safe">✓ NORMAL — {conf*100:.1f}%</div>', unsafe_allow_html=True)
        elif imgs:
            st.warning(f"Upload exactly 10 frames. You uploaded {len(imgs)}.")

    with tab_live:
        st.markdown("**Live Webcam Behavior Analysis**")
        st.info(f"Captures {SEQ_LEN} frames from webcam and classifies behavior in real-time.")
        interval = st.slider("Capture interval (seconds between frames)", 0.05, 0.5, 0.1, 0.05)

        run_live_beh = st.checkbox("▶ Start Live Analysis", key="live_behavior")
        frame_ph  = st.empty()
        result_ph = st.empty()

        if run_live_beh:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam.")
            else:
                while st.session_state.get("live_behavior", False):
                    frames = []
                    for _ in range(SEQ_LEN):
                        ret, frame = cap.read()
                        if ret:
                            frames.append(cv2.resize(frame, (224, 224)))
                        time.sleep(interval)
                    if len(frames) == SEQ_LEN:
                        label, conf, _ = run_behavior_analysis(frames, model)
                        # show last frame with overlay
                        disp = frames[-1].copy()
                        color = (0, 0, 255) if label == "fighting" else (0, 230, 118)
                        cv2.putText(disp, f"{label.upper()} {conf*100:.1f}%",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        frame_ph.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_container_width=True)
                        if label == "fighting":
                            result_ph.markdown('<div class="alert-danger">🚨 VIOLENT BEHAVIOUR</div>', unsafe_allow_html=True)
                        else:
                            result_ph.markdown('<div class="alert-safe">✓ NORMAL BEHAVIOUR</div>', unsafe_allow_html=True)
                cap.release()

# ═══════════════════════════════════════════════
# ── BIOMETRIC VERIFICATION ──
# ═══════════════════════════════════════════════
elif module == "👤 Biometric Verification":
    st.markdown('<h1>👤 BIOMETRIC VERIFICATION</h1>', unsafe_allow_html=True)
    st.markdown("FN13 + MTCNN — 1.56 MB model · 57× smaller than FaceNet", unsafe_allow_html=True)

    model = load_face_model()
    if model is None:
        st.markdown('<div class="alert-warning">⚠ Place <code>models/fn13_face.pth</code> in the models/ folder.</div>', unsafe_allow_html=True)
        st.stop()

    tab_upload, tab_live = st.tabs(["🖼 Upload Faces", "📷 Live Webcam"])

    with tab_upload:
        use_mtcnn = st.checkbox("Use MTCNN face alignment", value=True)
        eer_thr   = float(METRICS.get("biometric_fn13", {}).get("per_run", [{}])[0].get("EER_threshold", 0.155))
        threshold = st.slider("Similarity Threshold (EER ≈ 0.155)", 0.0, 1.0, eer_thr, 0.005)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**PROBE**")
            probe_file = st.file_uploader("Upload probe face", type=["jpg","jpeg","png"], key="probe")
        with col2:
            st.markdown("**GALLERY**")
            gallery_file = st.file_uploader("Upload gallery face", type=["jpg","jpeg","png"], key="gallery")

        if probe_file and gallery_file:
            probe_gray,   probe_orig   = preprocess_face_image(probe_file,   use_mtcnn)
            gallery_gray, gallery_orig = preprocess_face_image(gallery_file, use_mtcnn)
            c1, c2 = st.columns(2)
            with c1:
                st.image(cv2.cvtColor(probe_orig, cv2.COLOR_BGR2RGB), caption="Probe", use_container_width=True)
            with c2:
                st.image(cv2.cvtColor(gallery_orig, cv2.COLOR_BGR2RGB), caption="Gallery", use_container_width=True)

            if st.button("🔍 Verify Identity"):
                with st.spinner("Extracting embeddings..."):
                    emb_p = get_face_embedding(probe_gray, model)
                    emb_g = get_face_embedding(gallery_gray, model)
                    sim   = cosine_similarity(emb_p, emb_g)
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Cosine Similarity", f"{sim:.4f}")
                m2.metric("EER Threshold", f"{threshold:.4f}")
                m3.metric("Decision", "✅ SAME" if sim >= threshold else "❌ DIFFERENT")
                st.progress(min(float(sim), 1.0))
                if sim >= threshold:
                    st.markdown(f'<div class="alert-safe">✅ SAME PERSON — similarity {sim:.4f} ≥ {threshold:.4f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-danger">❌ DIFFERENT PERSON — similarity {sim:.4f} &lt; {threshold:.4f}</div>', unsafe_allow_html=True)

    with tab_live:
        st.markdown("**Live Face Verification via Webcam**")
        st.info("First enroll a reference face, then verify in real-time.")
        use_mtcnn_live = st.checkbox("Use MTCNN alignment", value=True, key="mtcnn_live")
        eer_thr_live   = float(METRICS.get("biometric_fn13", {}).get("per_run", [{}])[0].get("EER_threshold", 0.155))
        threshold_live = st.slider("Live Threshold", 0.0, 1.0, eer_thr_live, 0.005, key="live_thr")

        st.markdown("**Step 1 — Enroll Reference Face**")
        enroll_ph = st.empty()
        if st.button("📸 Capture Reference from Webcam"):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                st.session_state["ref_frame"] = frame
                enroll_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                caption="Reference captured", use_container_width=True)
                st.success("Reference face enrolled!")
            else:
                st.error("Could not capture from webcam.")

        if "ref_frame" in st.session_state:
            ref_gray = cv2.cvtColor(st.session_state["ref_frame"], cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.resize(ref_gray, (64, 64))
            ref_emb  = get_face_embedding(ref_gray, model)

            st.markdown("**Step 2 — Live Verification**")
            run_live_bio = st.checkbox("▶ Start Live Verification", key="live_bio")
            frame_ph2  = st.empty()
            result_ph2 = st.empty()

            if run_live_bio:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Cannot open webcam.")
                else:
                    while st.session_state.get("live_bio", False):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.resize(gray, (64, 64))
                        emb  = get_face_embedding(gray, model)
                        sim  = cosine_similarity(ref_emb, emb)
                        match = sim >= threshold_live
                        color = (0, 230, 118) if match else (0, 0, 255)
                        label_txt = f"MATCH {sim:.3f}" if match else f"NO MATCH {sim:.3f}"
                        disp = frame.copy()
                        cv2.putText(disp, label_txt, (10, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                        frame_ph2.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_container_width=True)
                        if match:
                            result_ph2.markdown('<div class="alert-safe">✅ IDENTITY VERIFIED</div>', unsafe_allow_html=True)
                        else:
                            result_ph2.markdown('<div class="alert-danger">❌ IDENTITY NOT VERIFIED</div>', unsafe_allow_html=True)
                    cap.release()
        else:
            st.warning("Capture a reference face first (Step 1).")

# ═══════════════════════════════════════════════
# ── UNIFIED THREAT ASSESSMENT ──
# ═══════════════════════════════════════════════
elif module == "🚨 Unified Threat":
    st.markdown('<h1>🚨 UNIFIED THREAT ASSESSMENT</h1>', unsafe_allow_html=True)
    st.markdown("All three models running together — combined threat score per frame/clip", unsafe_allow_html=True)

    m_weapon   = load_weapon_model()
    m_behavior = load_behavior_model()
    m_face     = load_face_model()

    any_loaded = any(m is not None for m in [m_weapon, m_behavior, m_face])
    if not any_loaded:
        st.markdown('<div class="alert-danger">No models found. Place model files in models/ folder.</div>', unsafe_allow_html=True)
        st.stop()

    # ── Weights & config ──────────────────────────
    st.markdown('<div class="module-header">⚙ THREAT SCORING CONFIG</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        w_weapon = st.slider("Weapon weight",   0.0, 1.0, 0.50, 0.05,
                             disabled=(m_weapon is None),
                             help="Contribution of weapon detection to threat score")
    with c2:
        w_behav  = st.slider("Behavior weight", 0.0, 1.0, 0.35, 0.05,
                             disabled=(m_behavior is None),
                             help="Contribution of violent behavior to threat score")
    with c3:
        w_face   = st.slider("Unknown face weight", 0.0, 1.0, 0.15, 0.05,
                             disabled=(m_face is None),
                             help="Contribution of unrecognized identity to threat score")

    # Normalize weights to sum to 1
    total_w = w_weapon + w_behav + w_face + 1e-9
    w_weapon /= total_w; w_behav /= total_w; w_face /= total_w

    conf_thr   = st.slider("YOLO confidence threshold", 0.01, 0.95, 0.25, 0.01)
    face_thr   = float(METRICS.get("biometric_fn13", {}).get("per_run", [{}])[0].get("EER_threshold", 0.155))
    eer_thr    = st.slider("Face similarity threshold (EER)", 0.0, 1.0, face_thr, 0.005)
    SEQ_LEN    = 10

    # ── Threat score helper ───────────────────────
    def compute_threat_score(weapon_score, behav_score, face_score):
        """All inputs 0–1. Returns weighted combined score 0–1."""
        return (w_weapon * weapon_score + w_behav * behav_score + w_face * face_score)

    def threat_level(score):
        if score >= 0.75: return "CRITICAL",  "#ff1744", "🔴"
        if score >= 0.50: return "HIGH",       "#ff6d00", "🟠"
        if score >= 0.25: return "MODERATE",   "#ffd740", "🟡"
        return                   "LOW",        "#00e676", "🟢"

    def draw_threat_overlay(frame, score, label, color_hex, icon,
                            weapon_s, behav_s, face_s):
        """Draw HUD overlay on a BGR frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        # Dark banner
        cv2.rectangle(overlay, (0, 0), (w, 90), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        # Threat bar background
        cv2.rectangle(frame, (10, 10), (w - 10, 30), (40, 40, 40), -1)
        bar_w = int((w - 20) * score)
        r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        cv2.rectangle(frame, (10, 10), (10 + bar_w, 30), (b, g, r), -1)
        # Text
        cv2.putText(frame, f"THREAT: {label}  {score*100:.0f}%",
                    (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (b, g, r), 2)
        cv2.putText(frame, f"W:{weapon_s*100:.0f}%  B:{behav_s*100:.0f}%  F:{face_s*100:.0f}%",
                    (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 220), 1)
        return frame

    # ── Enrolled reference face (optional) ───────
    st.markdown("---")
    st.markdown('<div class="module-header">👤 ENROLL REFERENCE FACE (optional)</div>', unsafe_allow_html=True)
    st.caption("If enrolled, unknown faces raise the threat score. Skip to ignore face component.")

    enroll_col1, enroll_col2 = st.columns([1, 2])
    with enroll_col1:
        enroll_file = st.file_uploader("Upload known/safe face", type=["jpg","jpeg","png"], key="enroll_unified")
        if st.button("📸 Capture from Webcam", key="cap_enroll"):
            cap_e = cv2.VideoCapture(0)
            ret_e, frame_e = cap_e.read()
            cap_e.release()
            if ret_e:
                st.session_state["unified_ref"] = frame_e
                st.success("Reference captured!")

    ref_emb_unified = None
    if enroll_file is not None:
        arr = np.frombuffer(enroll_file.read(), np.uint8)
        ref_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.resize(ref_gray, (64, 64))
        if m_face:
            ref_emb_unified = get_face_embedding(ref_gray, m_face)
        with enroll_col2:
            st.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB), caption="Enrolled reference", width=180)
    elif "unified_ref" in st.session_state:
        ref_img  = st.session_state["unified_ref"]
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.resize(ref_gray, (64, 64))
        if m_face:
            ref_emb_unified = get_face_embedding(ref_gray, m_face)
        with enroll_col2:
            st.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB), caption="Enrolled reference (webcam)", width=180)

    # ── Mode tabs ─────────────────────────────────
    st.markdown("---")
    tab_single, tab_video, tab_live = st.tabs(["🖼 Single Frame", "📹 Video Clip", "📷 Live Webcam"])

    # ─── Single Frame ────────────────────────────
    with tab_single:
        st.markdown("Upload one image — all three models run on it simultaneously.")
        up = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="unified_img")
        if up:
            arr     = np.frombuffer(up.read(), np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            weapon_score = behav_score = face_score = 0.0
            weapon_dets  = []
            behav_label  = "—"
            face_sim     = None

            with st.spinner("Running all models..."):
                # 1. Weapon
                if m_weapon:
                    _, dets, _, _ = run_weapon_detection(img_bgr, m_weapon, conf_thr)
                    weapon_score  = min(1.0, sum(d["conf"] for d in dets)) if dets else 0.0
                    weapon_dets   = dets

                # 2. Behavior — duplicate frame × SEQ_LEN (static image fallback)
                if m_behavior:
                    frames = [cv2.resize(img_bgr, (224, 224))] * SEQ_LEN
                    b_label, b_conf, _ = run_behavior_analysis(frames, m_behavior)
                    behav_score  = b_conf if b_label == "fighting" else 0.0
                    behav_label  = f"{b_label} ({b_conf*100:.1f}%)"

                # 3. Face
                if m_face:
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (64, 64))
                    emb  = get_face_embedding(gray, m_face)
                    if ref_emb_unified is not None:
                        sim       = cosine_similarity(emb, ref_emb_unified)
                        face_sim  = sim
                        face_score = max(0.0, 1.0 - sim / max(eer_thr, 0.01))
                        face_score = min(1.0, face_score)
                    else:
                        face_score = 0.0   # no reference → ignore

            threat = compute_threat_score(weapon_score, behav_score, face_score)
            lvl, col, icon = threat_level(threat)

            # Draw overlay
            disp = draw_threat_overlay(img_bgr.copy(), threat, lvl, col, icon,
                                       weapon_score, behav_score, face_score)

            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_b:
                st.markdown(f"""
                <div style="background:#0d1117;border:2px solid {col};border-radius:8px;padding:20px;text-align:center;margin-bottom:16px">
                    <div style="font-size:48px">{icon}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:28px;color:{col}">{lvl}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:42px;color:{col};font-weight:bold">{threat*100:.1f}%</div>
                    <div style="color:#607d8b;font-size:12px;margin-top:8px">COMBINED THREAT SCORE</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-card"><div class="label">🔫 Weapon Score</div>
                    <div class="value" style="font-size:20px">{weapon_score*100:.1f}%</div>
                    <div style="color:#546e7a;font-size:12px">{len(weapon_dets)} detection(s)</div></div>
                <div class="metric-card"><div class="label">🎬 Behavior Score</div>
                    <div class="value" style="font-size:20px">{behav_score*100:.1f}%</div>
                    <div style="color:#546e7a;font-size:12px">{behav_label}</div></div>
                <div class="metric-card"><div class="label">👤 Face Score</div>
                    <div class="value" style="font-size:20px">{face_score*100:.1f}%</div>
                    <div style="color:#546e7a;font-size:12px">{"sim=" + f"{face_sim:.3f}" if face_sim is not None else "no reference enrolled"}</div></div>
                """, unsafe_allow_html=True)

    # ─── Video Clip ──────────────────────────────
    with tab_video:
        st.markdown("Upload a video — each frame is scored, threat plotted over time.")
        vid_file = st.file_uploader("Upload video", type=["mp4","avi","mov"], key="unified_vid")
        sample_every = st.slider("Analyse every N frames", 1, 30, 10)

        if vid_file and st.button("▶ Run Unified Analysis on Video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(vid_file.read())
                tmp_path = tmp.name

            cap   = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            prog  = st.progress(0)
            frame_ph_v = st.empty()

            timeline = []   # list of (frame_idx, threat, weapon_s, behav_s, face_s)
            buffer   = []   # rolling SEQ_LEN frame buffer for behavior

            for fidx in range(total):
                ret, frame = cap.read()
                if not ret:
                    break
                buffer.append(cv2.resize(frame, (224, 224)))
                if len(buffer) > SEQ_LEN:
                    buffer.pop(0)

                if fidx % sample_every != 0:
                    continue

                weapon_score = behav_score = face_score = 0.0

                if m_weapon:
                    _, dets, _, _ = run_weapon_detection(frame, m_weapon, conf_thr)
                    weapon_score  = min(1.0, sum(d["conf"] for d in dets)) if dets else 0.0

                if m_behavior and len(buffer) == SEQ_LEN:
                    b_label, b_conf, _ = run_behavior_analysis(list(buffer), m_behavior)
                    behav_score = b_conf if b_label == "fighting" else 0.0

                if m_face and ref_emb_unified is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (64, 64))
                    emb  = get_face_embedding(gray, m_face)
                    sim  = cosine_similarity(emb, ref_emb_unified)
                    face_score = min(1.0, max(0.0, 1.0 - sim / max(eer_thr, 0.01)))

                threat = compute_threat_score(weapon_score, behav_score, face_score)
                lvl, col, icon = threat_level(threat)
                timeline.append((fidx, threat, weapon_score, behav_score, face_score))

                disp = draw_threat_overlay(frame.copy(), threat, lvl, col, icon,
                                           weapon_score, behav_score, face_score)
                frame_ph_v.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_container_width=True)
                prog.progress(min(fidx / max(total, 1), 1.0))

            cap.release()
            os.unlink(tmp_path)
            prog.progress(1.0)

            if timeline:
                st.markdown("---")
                st.markdown('<div class="module-header">📈 THREAT TIMELINE</div>', unsafe_allow_html=True)

                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 3), facecolor="#0a0c10")
                ax.set_facecolor("#0d1117")
                frames_t  = [t[0] for t in timeline]
                threats   = [t[1] for t in timeline]
                weapons   = [t[2] for t in timeline]
                behavs    = [t[3] for t in timeline]
                faces     = [t[4] for t in timeline]

                ax.fill_between(frames_t, threats, alpha=0.3, color="#ff1744")
                ax.plot(frames_t, threats, color="#ff1744", lw=2, label="Threat Score")
                ax.plot(frames_t, weapons, color="#00e5ff", lw=1, linestyle="--", label="Weapon")
                ax.plot(frames_t, behavs,  color="#ffd740", lw=1, linestyle="--", label="Behavior")
                ax.plot(frames_t, faces,   color="#e040fb", lw=1, linestyle="--", label="Face")

                for thr_val, label_txt, c in [(0.75,"CRITICAL","#ff1744"),(0.50,"HIGH","#ff6d00"),(0.25,"MODERATE","#ffd740")]:
                    ax.axhline(thr_val, color=c, alpha=0.4, lw=0.8, linestyle=":")
                    ax.text(frames_t[-1], thr_val + 0.01, label_txt, color=c, fontsize=7, ha="right")

                ax.set_xlim(frames_t[0], frames_t[-1])
                ax.set_ylim(0, 1.05)
                ax.set_xlabel("Frame", color="#607d8b")
                ax.set_ylabel("Score", color="#607d8b")
                ax.tick_params(colors="#607d8b")
                for spine in ax.spines.values():
                    spine.set_color("#00e5ff33")
                ax.legend(facecolor="#0d1117", labelcolor="#c8d6e5", fontsize=8)
                plt.tight_layout()

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_fig:
                    plt.savefig(tmp_fig.name, dpi=120, bbox_inches="tight", facecolor="#0a0c10")
                    st.image(tmp_fig.name, use_container_width=True)
                    os.unlink(tmp_fig.name)
                plt.close()

                # Summary stats
                peak   = max(threats)
                avg    = sum(threats) / len(threats)
                p_lvl, p_col, p_icon = threat_level(peak)
                a_lvl, a_col, a_icon = threat_level(avg)
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Peak Threat",    f"{peak*100:.1f}%")
                mc2.metric("Average Threat", f"{avg*100:.1f}%")
                mc3.metric("Peak Level",     f"{p_icon} {p_lvl}")
                mc4.metric("Frames Analysed", len(timeline))

    # ─── Live Webcam ─────────────────────────────
    with tab_live:
        st.markdown("**All three models — live, frame by frame**")
        st.info("Weapon detection runs every frame. Behavior runs on rolling 10-frame buffer. Face checked against enrolled reference.")

        interval_live = st.slider("Seconds between behavior analysis", 0.05, 0.5, 0.1, 0.05, key="unified_interval")
        run_unified   = st.checkbox("▶ Start Unified Live Assessment", key="live_unified")

        frame_ph_l   = st.empty()
        score_ph_l   = st.empty()
        history_ph_l = st.empty()

        if run_unified:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam.")
            else:
                buf         = []
                score_hist  = []   # rolling threat history for mini-chart

                while st.session_state.get("live_unified", False):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    buf.append(cv2.resize(frame, (224, 224)))
                    if len(buf) > SEQ_LEN:
                        buf.pop(0)

                    weapon_score = behav_score = face_score = 0.0

                    # Weapon — every frame
                    if m_weapon:
                        _, dets, _, _ = run_weapon_detection(frame, m_weapon, conf_thr)
                        weapon_score  = min(1.0, sum(d["conf"] for d in dets)) if dets else 0.0

                    # Behavior — when buffer full
                    if m_behavior and len(buf) == SEQ_LEN:
                        b_label, b_conf, _ = run_behavior_analysis(list(buf), m_behavior)
                        behav_score = b_conf if b_label == "fighting" else 0.0

                    # Face
                    if m_face and ref_emb_unified is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.resize(gray, (64, 64))
                        emb  = get_face_embedding(gray, m_face)
                        sim  = cosine_similarity(emb, ref_emb_unified)
                        face_score = min(1.0, max(0.0, 1.0 - sim / max(eer_thr, 0.01)))

                    threat = compute_threat_score(weapon_score, behav_score, face_score)
                    lvl, col, icon = threat_level(threat)
                    score_hist.append(threat)
                    if len(score_hist) > 60:
                        score_hist.pop(0)

                    # Draw HUD
                    disp = draw_threat_overlay(frame.copy(), threat, lvl, col, icon,
                                               weapon_score, behav_score, face_score)
                    frame_ph_l.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_container_width=True)

                    # Score card
                    score_ph_l.markdown(f"""
                    <div style="display:flex;gap:10px;margin:6px 0">
                        <div style="flex:1;background:#0d1117;border:2px solid {col};border-radius:6px;padding:10px;text-align:center">
                            <div style="font-family:'Share Tech Mono',monospace;font-size:22px;color:{col}">{icon} {lvl}</div>
                            <div style="font-family:'Share Tech Mono',monospace;font-size:32px;color:{col};font-weight:bold">{threat*100:.1f}%</div>
                        </div>
                        <div style="flex:1;background:#0d1117;border:1px solid #00e5ff33;border-radius:6px;padding:10px">
                            <div style="color:#607d8b;font-size:11px">🔫 WEAPON</div>
                            <div style="color:#00e5ff;font-size:18px;font-family:'Share Tech Mono',monospace">{weapon_score*100:.1f}%</div>
                            <div style="color:#607d8b;font-size:11px;margin-top:6px">🎬 BEHAVIOR</div>
                            <div style="color:#ffd740;font-size:18px;font-family:'Share Tech Mono',monospace">{behav_score*100:.1f}%</div>
                            <div style="color:#607d8b;font-size:11px;margin-top:6px">👤 FACE</div>
                            <div style="color:#e040fb;font-size:18px;font-family:'Share Tech Mono',monospace">{face_score*100:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    time.sleep(interval_live)
                cap.release()
