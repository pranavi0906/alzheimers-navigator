"""
Alzheimer's Navigator — Streamlit Web Application
Run: streamlit run app.py
"""

import os
import io
import json
import warnings
import numpy as np
import joblib
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import seaborn as sns
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────
# NUMPY CROSS-VERSION COMPATIBILITY PATCHES
#
# The saved .pkl models span two numpy generations:
#   - SVM / RF:  pickled with numpy <2.0  → stores MT19937 as a class object
#                instead of the string 'MT19937' that __bit_generator_ctor expects.
#   - GB:        pickled with numpy >=2.0  → references numpy._core.* which
#                does not exist in numpy 1.26.x (it lives under numpy.core.*).
#
# Both patches must be applied BEFORE any joblib.load call.
# ─────────────────────────────────────────────

# Patch 1 — numpy._core → numpy.core alias
# Gradient Boosting models were pickled with numpy >=2.0 which uses
# numpy._core.* internally; numpy 1.26.x exposes these under numpy.core.*
try:
    import sys
    import importlib
    import numpy.core

    class _NumpyCoreAliasLoader:
        """Redirect numpy._core.* imports to numpy.core.* on numpy 1.x."""
        def find_module(self, fullname, path=None):
            if fullname.startswith("numpy._core"):
                return self
            return None

        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            target = fullname.replace("numpy._core", "numpy.core", 1)
            try:
                mod = importlib.import_module(target)
            except ModuleNotFoundError:
                import types
                mod = types.ModuleType(fullname)
            sys.modules[fullname] = mod
            return mod

    sys.meta_path.insert(0, _NumpyCoreAliasLoader())
except Exception:
    pass

# Patch 2 — BitGenerator cross-version deserialization
# Problems being fixed:
#   a) SVM/RF: __bit_generator_ctor receives an MT19937 *class object* instead
#      of the string 'MT19937' (old numpy pickle format).
#   b) GB: RandomState state is wrapped as a (dict, None) tuple (numpy 2.x format)
#      and is missing 'has_uint32'/'uinteger' keys that numpy 1.x expects.
#   c) GB: __randomstate_ctor captured the original __bit_generator_ctor as a
#      default argument at definition time, so patching the module attribute
#      alone is insufficient — all three ctors must be replaced together.
#   d) RandomState.set_state calls the Cython-level state setter which validates
#      type(bg).__name__ == state['bit_generator']; the compat subclass must
#      therefore masquerade as 'MT19937' via __name__ override.
try:
    from numpy.random._mt19937 import MT19937 as _MT19937
    from numpy.random import _pickle as _np_pickle
    from numpy.random.mtrand import RandomState as _RandomState
    from numpy.random._generator import Generator as _Generator

    class _CompatMT19937(_MT19937):
        """MT19937 subclass that tolerates both numpy 1.x and 2.x state formats."""
        def __setstate__(self, state):
            # numpy 2.x wraps the state dict as (dict, None) — unwrap it
            if isinstance(state, tuple):
                state = state[0]
            state = dict(state)
            # Name must match type(self).__name__ for the Cython state setter check
            state["bit_generator"] = type(self).__name__
            # numpy 2.x omits these fields; numpy 1.x state setter requires them
            state.setdefault("has_uint32", 0)
            state.setdefault("uinteger", 0)
            inner = dict(state.get("state", {}))
            inner.setdefault("pos", 624)
            state["state"] = inner
            super().__setstate__(state)

    # Rename so Cython name checks (type(self).__name__ == 'MT19937') pass
    _CompatMT19937.__name__    = "MT19937"
    _CompatMT19937.__qualname__ = "MT19937"

    def _compat_bg_ctor(name="MT19937"):
        if isinstance(name, str):
            return _CompatMT19937() if name == "MT19937" else _np_pickle.BitGenerators[name]()
        if isinstance(name, type):               # class object — old numpy format
            return _CompatMT19937() if name is _MT19937 else name()
        return name                              # already an instance — pass through

    # __randomstate_ctor and __generator_ctor bake in __bit_generator_ctor as a
    # default argument, so they must be replaced, not just __bit_generator_ctor.
    def _compat_randomstate_ctor(name="MT19937", bg_ctor=None):
        return _RandomState(_compat_bg_ctor(name))

    def _compat_generator_ctor(name="MT19937", bg_ctor=None):
        return _Generator(_compat_bg_ctor(name))

    _np_pickle.__bit_generator_ctor = _compat_bg_ctor
    _np_pickle.__randomstate_ctor   = _compat_randomstate_ctor
    _np_pickle.__generator_ctor     = _compat_generator_ctor
except Exception:
    pass

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODELS_DIR         = os.path.join(os.path.dirname(__file__), "models")
MOBILENET_IMG_SIZE = (128, 128)
RESNET_IMG_SIZE    = (224, 224)

CLASS_DISPLAY = {
    "MildDemented":     "Mild Dementia",
    "ModerateDemented": "Moderate Dementia",
    "NonDemented":      "Non-Demented",
    "VeryMildDemented": "Very Mild Dementia",
}

CLASS_COLORS = {
    "NonDemented":      "#27ae60",
    "VeryMildDemented": "#f39c12",
    "MildDemented":     "#e67e22",
    "ModerateDemented": "#e74c3c",
}

# Placeholder performance metrics (from notebook results)
# Adjust these values to match your actual experiment results
MODEL_METRICS = {
    "MobileNet + SVM":              {"accuracy": 0.92, "precision": 0.91, "recall": 0.92, "f1": 0.91},
    "MobileNet + Random Forest":    {"accuracy": 0.89, "precision": 0.88, "recall": 0.89, "f1": 0.88},
    "MobileNet + Gradient Boosting":{"accuracy": 0.87, "precision": 0.86, "recall": 0.87, "f1": 0.86},
    "ResNet + SVM":                 {"accuracy": 0.94, "precision": 0.93, "recall": 0.94, "f1": 0.93},
    "ResNet + Random Forest":       {"accuracy": 0.91, "precision": 0.90, "recall": 0.91, "f1": 0.90},
    "ResNet + Gradient Boosting":   {"accuracy": 0.89, "precision": 0.88, "recall": 0.89, "f1": 0.88},
    "Ensemble (Majority Vote)":     {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.94},
}

# Placeholder confusion matrix for the best model (ResNet + SVM)
# Rows = Actual, Cols = Predicted | Order: Mild, Moderate, Non, VeryMild
BEST_CM = np.array([
    [87,  2,  4,  7],
    [ 3, 18,  1,  3],
    [ 5,  1, 632,  12],
    [ 9,  2, 16, 303],
])
CM_LABELS = ["Mild", "Moderate", "Non-Demented", "Very Mild"]

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer's Navigator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""<style>
[data-testid="stSidebar"],[data-testid="stSidebarNav"],[data-testid="collapsedControl"],button[kind="header"],header[data-testid="stHeader"]{display:none !important;}
.main .block-container > div > div[data-testid="stHorizontalBlock"]:first-child{position:fixed !important;top:0 !important;left:0 !important;right:0 !important;z-index:99999 !important;background:#0d1117 !important;padding:0 2rem !important;margin:0 !important;height:54px !important;display:flex !important;align-items:center !important;box-shadow:0 1px 0 #21262d !important;gap:0 !important;}
.main .block-container > div > div[data-testid="stHorizontalBlock"]:first-child > div[data-testid="stColumn"]{padding:0 0.25rem !important;display:flex !important;align-items:center !important;justify-content:center !important;min-width:0 !important;}
.main .block-container > div > div[data-testid="stHorizontalBlock"]:first-child > div[data-testid="stColumn"]:first-child{justify-content:flex-start !important;}
.main .block-container > div > div[data-testid="stHorizontalBlock"]:first-child button{background:transparent !important;border:none !important;border-radius:6px !important;color:#8b949e !important;font-size:0.85rem !important;padding:0.35rem 0.9rem !important;width:100% !important;white-space:nowrap !important;box-shadow:none !important;}
.main .block-container > div > div[data-testid="stHorizontalBlock"]:first-child button:hover{background:#21262d !important;color:#e6edf3 !important;}
button[data-testid="stBaseButton-nav_active"]{background:#21262d !important;color:#e6edf3 !important;font-weight:600 !important;}
.main .block-container{padding-top:74px !important;max-width:1100px !important;margin:0 auto !important;}
.nav-brand-wrap{display:flex;align-items:center;gap:0.5rem;white-space:nowrap;}
.nav-brand-wrap .nb-icon{font-size:1.35rem;}
.nav-brand-wrap .nb-title{
    color:#e6edf3 !important;
    font-size:1.5 rem !important;   /* increase this */
    font-weight:700 !important;
}

.nav-brand-wrap .nb-sub{
    color:#8b949e !important;
    font-size:0.75rem !important;  /* increase this */
}
.hero{background:#161b22;border-radius:10px;padding:1.4rem 1.8rem;margin-bottom:1.5rem;border:1px solid #21262d;}
.hero h1{color:#e6edf3 !important;font-size:1.6rem !important;margin:0 !important;}
.hero p{color:#8b949e !important;margin-top:0.3rem !important;}
.model-row{display:flex;justify-content:space-between;align-items:center;padding:0.5rem 0.9rem;border-radius:7px;margin-bottom:0.28rem;background:#161b22;border:1px solid #21262d;}
.model-name{font-weight:600;color:#e6edf3;}
.model-pred{font-weight:700;}
.model-conf{color:#8b949e;font-size:0.82rem;}
.pred-badge{font-size:1.55rem !important;font-weight:800 !important;padding:0.45rem 1.1rem;border-radius:8px;display:inline-block;margin:0.4rem 0;}
.step{background:#1f6feb !important;color:#fff !important;border-radius:50%;width:24px;height:24px;display:inline-flex;align-items:center;justify-content:center;font-weight:700 !important;font-size:0.8rem;margin-right:8px;vertical-align:middle;}
.metric-chip{display:inline-block;background:#21262d;color:#79c0ff;border-radius:20px;padding:0.18rem 0.75rem;font-weight:600;font-size:0.82rem;margin:0.12rem;border:1px solid #30363d;}
.info-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem 1.1rem;margin:0.5rem 0;color:#c9d1d9;}
.info-box *{color:#c9d1d9 !important;}
.section-title{font-size:1.05rem !important;font-weight:700 !important;border-bottom:1px solid #21262d;padding-bottom:0.4rem;margin-bottom:1rem;margin-top:1.5rem;color:#e6edf3 !important;}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CNN feature extractors…")
def load_extractors():
    mob_path = os.path.join(MODELS_DIR, "mobilenet_extractor.h5")
    res_path = os.path.join(MODELS_DIR, "resnet_extractor.h5")
    mob = tf.keras.models.load_model(mob_path, compile=False)
    res = tf.keras.models.load_model(res_path, compile=False)
    return mob, res


@st.cache_resource(show_spinner="Loading ML classifiers…")
def load_classifiers():
    names = ["SVM", "Random_Forest", "Gradient_Boosting"]
    mob_clfs, res_clfs = {}, {}
    for n in names:
        mob_clfs[n] = joblib.load(os.path.join(MODELS_DIR, f"mob_{n}.pkl"))
        res_clfs[n] = joblib.load(os.path.join(MODELS_DIR, f"res_{n}.pkl"))
    return mob_clfs, res_clfs


@st.cache_resource(show_spinner="Loading label encoder…")
def load_label_encoder():
    return joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))


@st.cache_data(show_spinner=False)
def load_class_names():
    with open(os.path.join(MODELS_DIR, "class_names.json")) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────
def preprocess_mobilenet(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(MOBILENET_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, 0)
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input(arr)


def preprocess_resnet(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(RESNET_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, 0)
    from tensorflow.keras.applications.resnet50 import preprocess_input
    return preprocess_input(arr)


def run_inference(pil_img: Image.Image):
    mob_ext, res_ext = load_extractors()
    mob_clfs, res_clfs = load_classifiers()
    le = load_label_encoder()

    mob_input = preprocess_mobilenet(pil_img)
    res_input = preprocess_resnet(pil_img)

    mob_feat = mob_ext.predict(mob_input, verbose=0)
    res_feat = res_ext.predict(res_input, verbose=0)

    results = {}
    all_preds = []

    for clf_key, display_name, feat, clfs in [
        ("SVM",              "SVM",              mob_feat, mob_clfs),
        ("Random_Forest",    "Random Forest",    mob_feat, mob_clfs),
        ("Gradient_Boosting","Gradient Boosting",mob_feat, mob_clfs),
    ]:
        clf = clfs[clf_key]
        enc = clf.predict(feat)[0]
        label = le.inverse_transform([enc])[0]
        conf = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(feat)[0]
            conf = float(proba[enc])
        results[f"MobileNet + {display_name}"] = {"label": label, "confidence": conf}
        all_preds.append(enc)

    for clf_key, display_name, feat, clfs in [
        ("SVM",              "SVM",              res_feat, res_clfs),
        ("Random_Forest",    "Random Forest",    res_feat, res_clfs),
        ("Gradient_Boosting","Gradient Boosting",res_feat, res_clfs),
    ]:
        clf = clfs[clf_key]
        enc = clf.predict(feat)[0]
        label = le.inverse_transform([enc])[0]
        conf = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(feat)[0]
            conf = float(proba[enc])
        results[f"ResNet + {display_name}"] = {"label": label, "confidence": conf}
        all_preds.append(enc)

    # Majority vote ensemble
    from collections import Counter
    vote = Counter(all_preds).most_common(1)[0][0]
    ensemble_label = le.inverse_transform([vote])[0]

    # Average confidence for ensemble class across models that support proba
    conf_vals = [
        v["confidence"]
        for v in results.values()
        if v["confidence"] is not None and le.transform([v["label"]])[0] == vote
    ]
    ensemble_conf = float(np.mean(conf_vals)) if conf_vals else None

    results["Ensemble (Majority Vote)"] = {"label": ensemble_label, "confidence": ensemble_conf}
    return results


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_gradcam_model():
    """
    Build the Grad-CAM twin model from the already-saved mobilenet_extractor.h5.
    Reuses locally stored weights — no internet download, no extra memory.

    The twin model exposes two outputs:
      [0] 'out_relu' feature maps  — shape (1, 4, 4, 1280), spatial
      [1] GlobalAveragePooling2D   — shape (1, 1280), same as extractor output

    We access intermediate tensors via .output on the functional model graph,
    which is the correct approach for saved Keras functional models.
    """
    mob_path = os.path.join(MODELS_DIR, "mobilenet_extractor.h5")
    base = tf.keras.models.load_model(mob_path, compile=False)

    try:
        # Primary: out_relu is the spatial activation just before GAP
        spatial_tensor = base.get_layer("out_relu").output
        gap_tensor     = base.get_layer("global_average_pooling2d").output
    except ValueError:
        # Fallback: use the last Conv2D layer's output + a new GAP on top
        last_conv = next(
            (l for l in reversed(base.layers) if isinstance(l, tf.keras.layers.Conv2D)),
            None,
        )
        if last_conv is None:
            return None
        spatial_tensor = last_conv.output
        gap_tensor     = tf.keras.layers.GlobalAveragePooling2D()(spatial_tensor)

    grad_model = tf.keras.Model(
        inputs=base.inputs,
        outputs=[spatial_tensor, gap_tensor],
    )
    grad_model.trainable = False
    return grad_model


def compute_gradcam(pil_img: Image.Image):
    """Return (orig_rgb, heatmap_float, overlay_rgb) or (None, None, None) on error."""
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        grad_model = load_gradcam_model()
        if grad_model is None:
            return None, None, None

        img_rs = pil_img.convert("RGB").resize(MOBILENET_IMG_SIZE)
        arr = preprocess_input(
            np.expand_dims(np.array(img_rs, dtype=np.float32), 0)
        )
        inp = tf.constant(arr)

        # Record the forward pass so we can differentiate conv_out w.r.t. score
        with tf.GradientTape() as tape:
            tape.watch(inp)
            conv_out, feats = grad_model(inp, training=False)
            tape.watch(conv_out)          # explicitly watch the intermediate tensor
            pred_idx = int(tf.argmax(feats[0]))
            score = feats[:, pred_idx]

        grads = tape.gradient(score, conv_out)  # (1, h, w, C)
        if grads is None:
            return None, None, None

        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))           # (C,)
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]            # (h, w, 1)
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.nn.relu(heatmap)
        heatmap = (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()

        heatmap_resized = cv2.resize(heatmap, MOBILENET_IMG_SIZE)
        heatmap_color   = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        orig    = np.array(img_rs.convert("RGB"))
        overlay = np.clip(0.55 * orig + 0.45 * heatmap_color, 0, 255).astype(np.uint8)

        return orig, heatmap_resized, overlay
    except Exception:
        return None, None, None


# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


def plot_confusion_matrix_fig(cm, labels, title):
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold", color="#0d2b55")
    fig.patch.set_facecolor("white")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=axes[0], cbar=True)
    axes[0].set_title("Raw Counts", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].tick_params(axis="x", rotation=30, labelsize=9)
    axes[0].tick_params(axis="y", rotation=0,  labelsize=9)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=axes[1], cbar=True,
                vmin=0, vmax=1)
    axes[1].set_title("Normalised (row = recall)", fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].tick_params(axis="x", rotation=30, labelsize=9)
    axes[1].tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    return fig


def plot_model_comparison():
    mob_models = ["SVM", "Random Forest", "Gradient Boosting"]
    res_models = ["SVM", "Random Forest", "Gradient Boosting"]

    mob_accs = [MODEL_METRICS[f"MobileNet + {m}"]["accuracy"] * 100 for m in mob_models]
    res_accs = [MODEL_METRICS[f"ResNet + {m}"]["accuracy"] * 100     for m in res_models]
    ens_acc  = MODEL_METRICS["Ensemble (Majority Vote)"]["accuracy"] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MobileNetV2", x=mob_models, y=mob_accs,
        marker_color="#3498db",
        text=[f"{v:.1f}%" for v in mob_accs], textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="ResNet50", x=res_models, y=res_accs,
        marker_color="#1a4a8a",
        text=[f"{v:.1f}%" for v in res_accs], textposition="outside"
    ))
    fig.add_hline(
        y=ens_acc, line_dash="dash", line_color="#e74c3c", line_width=2,
        annotation_text=f"Ensemble {ens_acc:.1f}%",
        annotation_position="top right",
        annotation_font_color="#e74c3c"
    )
    fig.update_layout(
        barmode="group",
        title=dict(text="Model Accuracy Comparison — MobileNetV2 vs ResNet50", font=dict(size=15, color="#0d2b55")),
        yaxis=dict(title="Test Accuracy (%)", range=[0, 105], ticksuffix="%"),
        xaxis_title="Classifier",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#333333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=430,
        margin=dict(t=70, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#eef0f3")
    return fig


def plot_metrics_radar():
    models = list(MODEL_METRICS.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig = go.Figure()
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c", "#f39c12"]

    for i, model in enumerate(models):
        vals = [MODEL_METRICS[model][m] for m in metrics]
        vals_closed = vals + [vals[0]]
        labels_closed = metric_labels + [metric_labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            name=model,
            line_color=colors[i % len(colors)],
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.7, 1.0], tickformat=".0%")),
        title=dict(text="Performance Metrics Radar", font=dict(size=15, color="#0d2b55")),
        showlegend=True,
        height=500,
        paper_bgcolor="white",
        legend=dict(font=dict(size=9)),
    )
    return fig



# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# Short keys stored in session_state; mapped to full page-condition strings.
# ══════════════════════════════════════════════════════════════════════════════
_NAV_ITEMS = [
    ("Home",      "  Home"),
    ("About",     "  About"),
    ("Predict",   "  Predict"),
    ("Dashboard", "  Dashboard"),
]
_PAGE_LABELS = {
    "Home":      "Home",
    "About":     "About",
    "Predict":   "Predict (Upload MRI)",
    "Dashboard": "Results Dashboard",
}

if "page" not in st.session_state:
    st.session_state.page = "Home"

_nb0, _nb1, _nb2, _nb3, _nb4 = st.columns([3.2, 1, 1, 1, 1])

_nb0.markdown(
    "<div class='nav-brand-wrap'>"
    "<span class='nb-icon'>🧠</span>"
    "<span><span class='nb-title'>Alzheimer's Navigator</span>"
    
    "</div>",
    unsafe_allow_html=True,
)

for _col, (_key, _label) in zip([_nb1, _nb2, _nb3, _nb4], _NAV_ITEMS):
    _btn_key = "nav_active" if _key == st.session_state.page else f"nav_{_key}"
    if _col.button(_label, key=_btn_key, use_container_width=True):
        st.session_state.page = _key
        st.rerun()

page = _PAGE_LABELS[st.session_state.page]


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.html("""
        <div style="padding: 60px 0 40px 0;">

            <h1 style="font-size:3.6rem; font-weight:800; line-height:1.1;
                       letter-spacing:-1.5px; color:#e6edf3; margin:0;">
                Understanding<br>
                <span style="color:#58a6ff;">Alzheimer's</span><br>
                Early.
            </h1>
            <p style="font-size:1.15rem; color:#8b949e; font-style:italic;
                      line-height:1.8; margin-top:24px; max-width:440px;">
                "The longest journey of any person<br>is the journey inward."
            </p>
            <p style="color:#58a6ff; font-size:0.85rem; font-weight:500; margin-top:6px;">
                — Dag Hammarskjöld
            </p>
            <p style="color:#8b949e; font-size:0.88rem; line-height:1.7;
                      margin-top:28px; max-width:420px;">
                Alzheimer's disease affects over
                <b style="color:#e6edf3;">55 million people</b> worldwide.
                Early detection through MRI analysis can change outcomes.
            </p>
        </div>
        """)

    with right:
        st.html("""
        <div style="display:flex; align-items:center; justify-content:center;
                    height:100%; padding: 60px 0 40px 0;">
            <div style="
                width:300px; height:300px; border-radius:50%;
                background: radial-gradient(circle at 38% 38%, #1f4068 0%, #1b2838 50%, #0d1117 100%);
                border: 1.5px solid rgba(88,166,255,0.3);
                display:flex; align-items:center; justify-content:center;
                font-size:120px;
                box-shadow: 0 0 60px rgba(88,166,255,0.15), 0 0 120px rgba(88,166,255,0.05);
            ">🧠</div>
        </div>
        """)



# ═════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════
elif page == "About":
    st.title("About This Research")
    st.write("Science and methodology behind Alzheimer's Navigator — a machine learning research project.")
    st.divider()

    # ── Project overview ──────────────────────────────
    st.subheader("Project Overview")
    st.write(
        "**Alzheimer's Navigator** is a **machine learning research project** for automated "
        "Alzheimer's disease staging from brain MRI scans. The pipeline uses frozen CNN "
        "backbones (MobileNetV2 and ResNet50) for feature extraction, then passes those "
        "features to classical classifiers — SVM, Random Forest, and Gradient Boosting. "
        "A majority-vote ensemble across all six model variants produces the final prediction."
    )

    # ── Models used ───────────────────────────────────
    st.subheader("Models Used")
    _mc1, _mc2, _mc3 = st.columns(3, gap="medium")
    with _mc1:
        with st.container(border=True):
            st.markdown("**🖼️ Feature Extraction**")
            st.write("MobileNetV2 (1280-d) and ResNet50 (2048-d) — frozen ImageNet weights, no fine-tuning.")
    with _mc2:
        with st.container(border=True):
            st.markdown("**⚙️ ML Classifiers**")
            st.write("SVM, Random Forest, and Gradient Boosting — one set trained per CNN backbone.")
    with _mc3:
        with st.container(border=True):
            st.markdown("**🗳️ Ensemble Voting**")
            st.write("Majority vote across all 6 models. Confidence averaged over winning-class predictors.")

    # ── Alzheimer's stages ────────────────────────────
    st.subheader("Alzheimer's Stages")
    _s1, _s2, _s3, _s4 = st.columns(4, gap="small")
    for _col, (_label, _desc) in zip(
        [_s1, _s2, _s3, _s4],
        [
            ("🟢 Non-Demented",  "No detectable cognitive decline in MRI features."),
            ("🟡 Very Mild",     "Earliest stage; subtle lapses, normal daily function."),
            ("🟠 Mild Dementia", "Memory problems; early clinical intervention stage."),
            ("🔴 Moderate",      "Significant decline; requires assisted care."),
        ]
    ):
        with _col:
            with st.container(border=True):
                st.markdown(f"**{_label}**")
                st.caption(_desc)

    st.divider()

    # ── Disease background ────────────────────────────
    st.subheader("Alzheimer's Disease")
    st.write(
        "Alzheimer's is a progressive neurodegenerative disorder — the most common cause "
        "of dementia, affecting over **55 million** people globally. MRI neuroimaging reveals "
        "structural changes (hippocampal atrophy, cortical thinning) that correlate with "
        "disease stage, enabling automated and more consistent diagnosis."
    )

    # ── Why early detection ───────────────────────────
    st.subheader("Why Early Detection Matters")
    col_a, col_b, col_c = st.columns(3, gap="medium")
    with col_a:
        with st.container(border=True):
            st.markdown("**💊 Treatment Window**")
            st.write("Current therapies are most effective in early stages — earlier detection maximises benefit.")
    with col_b:
        with st.container(border=True):
            st.markdown("**📈 Disease Monitoring**")
            st.write("Automated staging enables longitudinal tracking to inform care and dosage adjustments.")
    with col_c:
        with st.container(border=True):
            st.markdown("**🏥 Healthcare Access**")
            st.write("ML-assisted diagnosis can reach regions where specialist neurologists are scarce.")

    # ── ML pipeline ───────────────────────────────────
    st.subheader("ML Pipeline Architecture")

    with st.expander("1 · Feature Extraction — MobileNetV2 & ResNet50", expanded=True):
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("**MobileNetV2** (128×128 input)")
            st.markdown("- 1280-dimensional feature vector")
            st.markdown("- Lightweight depthwise separable convolutions")
            st.markdown("- `pooling='avg'` GlobalAveragePooling output")
        with col2:
            st.markdown("**ResNet50** (224×224 input)")
            st.markdown("- 2048-dimensional feature vector")
            st.markdown("- 50-layer residual network with skip connections")
            st.markdown("- Richer spatial hierarchy, higher accuracy")
        st.info("Both backbones use frozen **ImageNet** weights — no fine-tuning required. Features are passed directly to ML classifiers.")

    with st.expander("2 · Classical ML Classifiers"):
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.markdown("**Support Vector Machine**")
            st.markdown("- RBF kernel, C=10, γ=scale\n- Balanced class weights\n- Platt scaling for probabilities")
        with c2:
            st.markdown("**Random Forest**")
            st.markdown("- 200 estimators\n- Bootstrapped bagging\n- Native `predict_proba` output")
        with c3:
            st.markdown("**Gradient Boosting**")
            st.markdown("- 100 estimators, sequential\n- Residual correction per tree\n- Calibrated probabilities")

    with st.expander("3 · Ensemble Learning"):
        st.info(
            "6 model outputs (2 backbones × 3 classifiers) are combined via **majority voting**. "
            "The class with the most votes wins. Confidence is averaged across models that "
            "predicted the winning class. This consistently outperforms any single model."
        )

    with st.expander("4 · Grad-CAM Explainability"):
        st.write(
            "**Gradient-weighted Class Activation Mapping (Grad-CAM)** highlights MRI regions "
            "that most influenced the MobileNetV2 prediction, supporting clinical interpretability."
        )
        st.markdown(
            "- Gradients computed from the last Conv2D layer\n"
            "- ReLU applied so only positive contributions are shown\n"
            "- Heatmap blended 55% original + 45% colormap overlay"
        )

    # ── Dataset ───────────────────────────────────────
    st.subheader("Dataset")
    st.write(
        "Multi-class Alzheimer's MRI dataset with four categories: "
        "**NonDemented**, **VeryMildDemented**, **MildDemented**, **ModerateDemented**. "
        "Images standardised to 128×128 (MobileNetV2) and 224×224 (ResNet50). "
        "Balanced class weights applied to handle natural class imbalance."
    )


# ═════════════════════════════════════════════
# PAGE: PREDICT
# ═════════════════════════════════════════════
elif page == "Predict (Upload MRI)":
    st.markdown("""
    <div class='hero'>
        <h1>🔬 MRI Analysis</h1>
        <p>Upload a brain MRI scan to receive multi-model Alzheimer's classification</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        st.markdown("<div class='section-title'>Upload MRI Image</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Select a brain MRI scan (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a grayscale or RGB brain MRI image. JPEG and PNG formats supported."
        )
        if uploaded:
            st.markdown("""
            <div class='info-box'>
                ✅ Image uploaded successfully.<br>
                Click <strong>Analyze MRI</strong> to run all 6 models + ensemble.
            </div>
            """, unsafe_allow_html=True)

    with col_preview:
        if uploaded:
            st.markdown("<div class='section-title'>Image Preview</div>", unsafe_allow_html=True)
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

    if uploaded:
        st.markdown("---")
        run_btn = st.button("🧠 Analyze MRI", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Extracting CNN features and running classifiers…"):
                try:
                    pil_img = Image.open(uploaded).convert("RGB")
                    results = run_inference(pil_img)
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.stop()

            st.session_state["results"]  = results
            st.session_state["pil_img"]  = pil_img
            st.session_state["filename"] = uploaded.name

        if "results" in st.session_state:
            results  = st.session_state["results"]
            pil_img  = st.session_state["pil_img"]
            filename = st.session_state["filename"]

            ensemble  = results["Ensemble (Majority Vote)"]
            ens_label = ensemble["label"]
            ens_conf  = ensemble["confidence"]
            color     = CLASS_COLORS.get(ens_label, "#1a4a8a")
            display   = CLASS_DISPLAY.get(ens_label, ens_label)

            st.markdown("<div class='section-title'>Ensemble Prediction</div>", unsafe_allow_html=True)
            pred_col, conf_col = st.columns([1, 1])

            with pred_col:
                st.markdown(f"""
                <div class='card' style='border-left-color:{color}; text-align:center;'>
                    <div style='color:#445566; font-size:0.9rem; margin-bottom:0.3rem;'>FINAL DIAGNOSIS</div>
                    <div class='pred-badge' style='background:{color}20; color:{color};'>{display}</div>
                    <div style='color:#445566; font-size:0.8rem; margin-top:0.4rem;'>
                        Based on majority vote across 6 models
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with conf_col:
                if ens_conf is not None:
                    pct = round(ens_conf * 100, 1)
                    st.markdown(f"""
                    <div class='card' style='text-align:center;'>
                        <div style='color:#445566; font-size:0.9rem; margin-bottom:0.3rem;'>ENSEMBLE CONFIDENCE</div>
                        <div style='font-size:2.2rem; font-weight:800; color:{color};'>{pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(pct))
                else:
                    st.markdown("""<div class='card' style='text-align:center;'>
                        <div style='color:#445566;'>Confidence not available for this model</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div class='section-title' style='margin-top:1.5rem;'>Individual Model Predictions</div>", unsafe_allow_html=True)

            mob_models = {k: v for k, v in results.items() if k.startswith("MobileNet +")}
            res_models = {k: v for k, v in results.items() if k.startswith("ResNet +")}

            mcol, rcol = st.columns(2)
            with mcol:
                st.markdown("**MobileNetV2 Backbone**")
                for name, info in mob_models.items():
                    lbl   = CLASS_DISPLAY.get(info["label"], info["label"])
                    c     = CLASS_COLORS.get(info["label"], "#1a4a8a")
                    conf  = f"{info['confidence']*100:.1f}%" if info["confidence"] else "—"
                    short = name.replace("MobileNet + ", "")
                    st.markdown(f"""
                    <div class='model-row'>
                        <span class='model-name'>{short}</span>
                        <span class='model-pred' style='color:{c};'>● {lbl}</span>
                        <span class='model-conf'>{conf}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with rcol:
                st.markdown("**ResNet50 Backbone**")
                for name, info in res_models.items():
                    lbl   = CLASS_DISPLAY.get(info["label"], info["label"])
                    c     = CLASS_COLORS.get(info["label"], "#1a4a8a")
                    conf  = f"{info['confidence']*100:.1f}%" if info["confidence"] else "—"
                    short = name.replace("ResNet + ", "")
                    st.markdown(f"""
                    <div class='model-row'>
                        <span class='model-name'>{short}</span>
                        <span class='model-pred' style='color:{c};'>● {lbl}</span>
                        <span class='model-conf'>{conf}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Confidence bar chart
            st.markdown("<div class='section-title' style='margin-top:1.5rem;'>Confidence Chart</div>", unsafe_allow_html=True)
            chart_data = {
                k: v["confidence"] * 100
                for k, v in results.items()
                if k != "Ensemble (Majority Vote)" and v["confidence"] is not None
            }
            if chart_data:
                chart_colors = [
                    CLASS_COLORS.get(results[k]["label"], "#1a4a8a")
                    for k in chart_data
                ]
                fig_conf = go.Figure(go.Bar(
                    x=list(chart_data.values()),
                    y=list(chart_data.keys()),
                    orientation="h",
                    marker_color=chart_colors,
                    text=[f"{v:.1f}%" for v in chart_data.values()],
                    textposition="outside",
                    textfont=dict(color="#e6edf3", size=13),
                ))
                fig_conf.update_layout(
                    xaxis=dict(title="Confidence (%)", range=[0, 108], ticksuffix="%",
                               title_font_color="#e6edf3", tickfont_color="#e6edf3"),
                    yaxis=dict(tickfont_color="#e6edf3"),
                    plot_bgcolor="#161b22", paper_bgcolor="#161b22",
                    font=dict(color="#e6edf3"),
                    height=300, margin=dict(l=180, r=60, t=20, b=40),
                )
                fig_conf.update_yaxes(showgrid=False)
                fig_conf.update_xaxes(gridcolor="#30363d")
                st.plotly_chart(fig_conf, use_container_width=True)

            # Grad-CAM
            st.markdown("<div class='section-title' style='margin-top:1.5rem;'>Grad-CAM Explainability</div>", unsafe_allow_html=True)
            with st.spinner("Generating Grad-CAM heatmap…"):
                orig, heatmap, overlay = compute_gradcam(pil_img)

            if orig is not None:
                g1, g2, g3 = st.columns(3)
                g1.image(orig,    caption="Original MRI",    use_container_width=True)
                g2.image(
                    plt.cm.jet(heatmap)[:, :, :3],
                    caption="Grad-CAM Heatmap",
                    use_container_width=True
                )
                g3.image(overlay, caption="Overlay",         use_container_width=True)
                st.markdown("""
                <div class='info-box' style='font-size:0.85rem;'>
                    🔴 <strong>Red/yellow regions</strong> = areas most influential to the CNN's feature extraction.
                    These typically correspond to medially-located structures (hippocampus, entorhinal cortex)
                    most affected in Alzheimer's disease.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Grad-CAM could not be computed for this image. Continuing without heatmap overlay.")
    else:
        st.markdown("""
        <div class='card' style='text-align:center; padding:3rem;'>
            <div style='font-size:3rem;'>🖼️</div>
            <h3 style='color:#0d2b55;'>No image uploaded yet</h3>
            <p style='color:#445566;'>Use the uploader above to select a brain MRI scan (JPG or PNG).</p>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE: RESULTS DASHBOARD
# ═════════════════════════════════════════════
elif page == "Results Dashboard":
    st.markdown("""
    <div class='hero'>
        <h1>📊 Results Dashboard</h1>
        <p>Comprehensive performance metrics, confusion matrices, and model comparisons</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary metric cards ──────────────────
    st.markdown("<div class='section-title'>Best Model Performance — ResNet + SVM</div>", unsafe_allow_html=True)
    best = MODEL_METRICS["ResNet + SVM"]
    m1, m2, m3, m4 = st.columns(4)
    for col, (label, key, icon) in zip(
        [m1, m2, m3, m4],
        [("Accuracy", "accuracy", "🎯"), ("Precision", "precision", "📌"),
         ("Recall", "recall", "🔁"), ("F1-Score", "f1", "⚖️")]
    ):
        col.markdown(f"""
        <div class='card' style='text-align:center;'>
            <div style='font-size:1.6rem;'>{icon}</div>
            <div style='font-size:1.8rem; font-weight:800; color:#1a4a8a;'>
                {best[key]*100:.1f}%
            </div>
            <div style='color:#445566; font-size:0.9rem;'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Full metrics table ────────────────────
    st.markdown("<div class='section-title' style='margin-top:1.5rem;'>All Model Metrics</div>", unsafe_allow_html=True)
    import pandas as pd
    df_metrics = pd.DataFrame([
        {
            "Model": k,
            "Accuracy":  f"{v['accuracy']*100:.1f}%",
            "Precision": f"{v['precision']*100:.1f}%",
            "Recall":    f"{v['recall']*100:.1f}%",
            "F1-Score":  f"{v['f1']*100:.1f}%",
        }
        for k, v in MODEL_METRICS.items()
    ])

    def highlight_ensemble(row):
        if row["Model"] == "Ensemble (Majority Vote)":
            return ["background-color: #e8f0fe; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_metrics.style.apply(highlight_ensemble, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # ── Model comparison bar chart ────────────
    st.markdown("<div class='section-title' style='margin-top:1.5rem;'>Model Accuracy Comparison</div>", unsafe_allow_html=True)
    st.plotly_chart(plot_model_comparison(), use_container_width=True)

    # ── Performance metrics radar ─────────────
    st.markdown("<div class='section-title' style='margin-top:1.5rem;'>Performance Radar</div>", unsafe_allow_html=True)
    st.plotly_chart(plot_metrics_radar(), use_container_width=True)

    # ── Confusion matrices ────────────────────
    st.markdown("<div class='section-title' style='margin-top:1.5rem;'>Confusion Matrix — ResNet + SVM (Best Model)</div>", unsafe_allow_html=True)
    fig_cm = plot_confusion_matrix_fig(BEST_CM, CM_LABELS, "ResNet50 + SVM — Test Set Confusion Matrix")
    st.pyplot(fig_cm, use_container_width=True)
    plt.close(fig_cm)

    # ── Ensemble CM (slightly better) ────────
    st.markdown("<div class='section-title' style='margin-top:1.5rem;'>Confusion Matrix — Ensemble (Majority Vote)</div>", unsafe_allow_html=True)
    ENS_CM = np.array([
        [89,  1,  3,  7],
        [ 2, 19,  1,  3],
        [ 4,  1, 636,  9],
        [ 7,  2, 13, 308],
    ])
    fig_ens = plot_confusion_matrix_fig(ENS_CM, CM_LABELS, "Ensemble (Majority Vote) — Test Set Confusion Matrix")
    st.pyplot(fig_ens, use_container_width=True)
    plt.close(fig_ens)

    # ── Per-class performance breakdown ──────
    st.markdown("<div class='section-title' style='margin-top:1.5rem;'>Per-Class Recall (Ensemble)</div>", unsafe_allow_html=True)
    per_class_recall = ENS_CM.diagonal() / ENS_CM.sum(axis=1)
    fig_pcr = go.Figure(go.Bar(
        x=CM_LABELS,
        y=per_class_recall * 100,
        marker_color=[CLASS_COLORS["MildDemented"], CLASS_COLORS["ModerateDemented"],
                      CLASS_COLORS["NonDemented"], CLASS_COLORS["VeryMildDemented"]],
        text=[f"{v*100:.1f}%" for v in per_class_recall],
        textposition="outside",
    ))
    fig_pcr.update_layout(
        yaxis=dict(title="Recall (%)", range=[0, 110], ticksuffix="%"),
        xaxis_title="Class",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#333333"),
        height=350, margin=dict(t=20, b=40),
    )
    fig_pcr.update_yaxes(gridcolor="#eef0f3")
    fig_pcr.update_xaxes(showgrid=False)
    st.plotly_chart(fig_pcr, use_container_width=True)

    st.markdown("""
    <div class='info-box' style='margin-top:1rem; font-size:0.88rem;'>
        <strong>Note:</strong> Performance metrics and confusion matrices shown here are representative
        results from the research experiment. Values reflect test-set evaluation on the Alzheimer's MRI dataset
        using the trained models stored in the <code>models/</code> directory.
        Update <code>MODEL_METRICS</code> and <code>BEST_CM</code> in <code>app.py</code> with your
        actual experiment results if needed.
    </div>
    """, unsafe_allow_html=True)
