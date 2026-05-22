"""Natural Language Autoencoder — Streamlit Dashboard.

Read-only viewer: loads pre-generated artifacts from data/, results/,
checkpoints/. Does NOT re-train the model.

Optional Page 5 ("Custom Text Demo") loads the SmolLM2 model on demand
for live activation extraction + cluster lookup.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
DATA_DIR    = ROOT / "data"
CKPT_DIR    = ROOT / "checkpoints"
CONFIG_PATH = ROOT / "config.yaml"

# ── helpers ────────────────────────────────────────────────────────────────
@st.cache_data
def load_config():
    from src.utils import load_config as _lc
    return _lc(CONFIG_PATH)

@st.cache_data
def load_eval_results():
    p = RESULTS_DIR / "eval_results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

@st.cache_data
def load_control():
    p = RESULTS_DIR / "control_shuffled.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

@st.cache_data
def load_per_cluster():
    p = RESULTS_DIR / "per_cluster_results.csv"
    return pd.read_csv(p, encoding="utf-8") if p.exists() else None

@st.cache_data
def load_qualitative():
    p = RESULTS_DIR / "qualitative_examples.csv"
    return pd.read_csv(p, encoding="utf-8") if p.exists() else None

@st.cache_data
def load_clusters_llm():
    p = DATA_DIR / "clusters_llm.json"
    if not p.exists():
        p = DATA_DIR / "clusters.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

@st.cache_data
def load_training_history():
    p = RESULTS_DIR / "training_history.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

@st.cache_data
def load_activations_meta():
    p = DATA_DIR / "activations.meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NL Autoencoder Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🧠 NL Autoencoder")
st.sidebar.caption("Small-scale reimplementation of Anthropic's NLA paper")

PAGES = [
    "Overview",
    "Metrics & Control",
    "Training Curve",
    "Plots",
    "Qualitative Examples",
    "Cluster Explanations",
    "Custom Text Demo",
]
page = st.sidebar.radio("Navigate", PAGES)

cfg = load_config()

# ── helpers for missing-data warnings ─────────────────────────────────────
def missing(label: str):
    st.warning(f"**{label}** not found — run the pipeline scripts first.", icon="⚠️")


# ── Custom Text Demo helper functions ──────────────────────────────────────
def compute_top_k_clusters(act_vec: np.ndarray, km, clusters_data: list, k: int = 3):
    """Return top-k nearest clusters sorted by Euclidean distance (ascending)."""
    centroids = km.cluster_centers_.astype(np.float64)
    v = act_vec.astype(np.float64).ravel()
    dists = np.linalg.norm(centroids - v, axis=1)
    top_idx = np.argsort(dists)[:k]
    result = []
    for rank, ci in enumerate(top_idx):
        card = next((c for c in clusters_data if c["cluster_id"] == int(ci)), {})
        result.append({
            "rank": rank + 1,
            "cluster_id": int(ci),
            "distance": float(dists[ci]),
            "size": card.get("size", "?"),
            "explanation": card.get("explanation", "—"),
            "representative_texts": card.get("representative_texts", []),
        })
    return result


def compute_assignment_confidence(top_clusters: list):
    """ratio = d1/d2; lower ratio = more confident."""
    if len(top_clusters) < 2:
        return {"ratio": None, "margin": None, "label": "UNKNOWN"}
    d1, d2 = top_clusters[0]["distance"], top_clusters[1]["distance"]
    if d2 == 0:
        return {"ratio": 0.0, "margin": 0.0, "label": "HIGH"}
    ratio  = d1 / d2
    margin = d2 - d1
    label  = "HIGH" if ratio < 0.85 else ("MEDIUM" if ratio < 0.95 else "LOW")
    return {"ratio": ratio, "margin": margin, "label": label}


def compute_ood_status(act_vec: np.ndarray, all_acts: np.ndarray, km, percentile: float = 90.0):
    """Is the custom input farther from training clusters than `percentile`% of training data?"""
    centroids   = km.cluster_centers_.astype(np.float64)
    train_dists = np.min(
        np.linalg.norm(
            all_acts.astype(np.float64)[:, None, :] - centroids[None, :, :], axis=2
        ),
        axis=1,
    )
    threshold    = float(np.percentile(train_dists, percentile))
    custom_dist  = float(np.min(np.linalg.norm(centroids - act_vec.astype(np.float64).ravel(), axis=1)))
    pct_rank     = float(np.mean(train_dists <= custom_dist) * 100)
    return {
        "custom_dist":    custom_dist,
        "threshold":      threshold,
        "percentile_rank": pct_rank,
        "is_ood":         custom_dist > threshold,
        "train_p50":      float(np.percentile(train_dists, 50)),
        "train_p90":      float(np.percentile(train_dists, 90)),
    }


def safe_single_example_metrics(act_vec: np.ndarray, recon_vec: np.ndarray):
    """MSE, RMSE, cosine, norm-ratio — all stable for a single sample. FVE excluded."""
    a, r   = act_vec.astype(np.float64).ravel(), recon_vec.astype(np.float64).ravel()
    mse    = float(np.mean((a - r) ** 2))
    norm_a = float(np.linalg.norm(a))
    norm_r = float(np.linalg.norm(r))
    denom  = norm_a * norm_r
    cosine = float(np.dot(a, r) / denom) if denom > 1e-12 else float("nan")
    return {
        "mse":                  mse,
        "rmse":                 float(np.sqrt(mse)),
        "cosine":               cosine,
        "norm_original":        norm_a,
        "norm_reconstruction":  norm_r,
        "norm_ratio":           (norm_r / norm_a) if norm_a > 1e-12 else float("nan"),
    }


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Natural Language Autoencoder — Small Model")
    st.markdown(
        """
        A small-scale reimplementation of
        [Anthropic's NLA paper](https://transformer-circuits.pub/2026/nla/index.html).

        **Pipeline:**
        ```
        Text dataset → Small LLM → Hidden activations
                                 → LLM verbalizer → English explanation
                                                   → MLP reconstructor → Reconstructed activation
                                                                        → FVE / MSE / Cosine
        ```
        The core insight: if a **natural-language description** of an activation can recover
        the activation to within a small residual, the description has captured something real.
        """
    )

    meta = load_activations_meta()
    d = cfg["data"]
    v = cfg["verbalizer"]
    r = cfg["reconstructor"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Model")
        st.markdown(f"**LLM:** `{cfg['model']['name']}`")
        st.markdown(f"**Layer:** {meta['extracted_layer'] if meta else cfg['model']['layer_index']} "
                    f"of {meta['num_hidden_layers'] if meta else '?'} (2/3 depth)")
        st.markdown(f"**Hidden size:** {meta['hidden_size'] if meta else '?'}")
        st.markdown(f"**Pooling:** {cfg['model']['pooling']}")
    with col2:
        st.subheader("Data")
        st.markdown(f"**Dataset:** `{d['dataset_name']} / {d['dataset_config']}`")
        st.markdown(f"**Samples:** {d['num_samples']}")
        st.markdown(f"**Min/max chars:** {d['min_chars']} / {d['max_chars']}")
        st.markdown(f"**Seed:** {cfg['seed']}")
    with col3:
        st.subheader("Experiment")
        st.markdown(f"**Clusters (K):** {v['num_clusters']}")
        st.markdown(f"**Reconstructor:** MLP {r['hidden_dims']}")
        st.markdown(f"**Train/val/test:** "
                    f"{int(d['num_samples']*(1-r['val_frac']-r['test_frac']))} / "
                    f"{int(d['num_samples']*r['val_frac'])} / "
                    f"{int(d['num_samples']*r['test_frac'])}")
        st.markdown(f"**Epochs:** {r['epochs']}")

    st.divider()
    st.subheader("config.yaml")
    st.code(CONFIG_PATH.read_text(encoding="utf-8"), language="yaml")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Metrics & Control
# ══════════════════════════════════════════════════════════════════════════
elif page == "Metrics & Control":
    st.title("Evaluation Metrics (held-out test set)")
    ev = load_eval_results()
    ctrl = load_control()

    if ev is None:
        missing("results/eval_results.json")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FVE (global)", f"{ev['fve_global']:.4f}")
    c2.metric("FVE per-sample (median)", f"{ev['fve_sample_median']:.4f}")
    c3.metric("Cosine sim. (mean)", f"{ev['cosine_mean']:.4f}")
    c4.metric("MSE (mean)", f"{ev['mse_mean']:.4f}")

    st.divider()
    st.subheader("Real vs Shuffled-Explanation Control")
    st.markdown(
        """
        The **shuffled control** trains an identical MLP with (explanation → activation) pairs
        randomly permuted. A large gap confirms the explanations — not the activation prior —
        drive the FVE.
        """
    )

    if ctrl:
        real_fve    = ctrl["real"]["fve_global"]
        shuf_fve    = ctrl["control_shuffled"]["fve_global"]
        delta       = ctrl["delta_fve_global"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Real FVE", f"{real_fve:.4f}")
        col2.metric("Shuffled FVE", f"{shuf_fve:.4f}")
        col3.metric("Δ (real − shuffled)", f"{delta:+.4f}",
                    delta_color="normal")

        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=["Real explanations", "Shuffled control"],
            y=[real_fve, shuf_fve],
            marker_color=["#3a78c2", "#c23a3a"],
            text=[f"{real_fve:.3f}", f"{shuf_fve:.3f}"],
            textposition="outside",
        ))
        fig.update_layout(
            title="Global FVE: Real vs Shuffled",
            yaxis_title="FVE",
            yaxis=dict(range=[min(shuf_fve - 0.1, -0.2), 1.05]),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full metric comparison")
        rows = []
        for k in ["fve_global", "fve_sample_mean", "fve_sample_median",
                  "mse_mean", "cosine_mean"]:
            rows.append({
                "Metric": k,
                "Real": round(ctrl["real"][k], 6),
                "Shuffled": round(ctrl["control_shuffled"][k], 6),
                "Δ": round(ctrl["real"][k] - ctrl["control_shuffled"][k], 6),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        missing("results/control_shuffled.json")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Training Curve
# ══════════════════════════════════════════════════════════════════════════
elif page == "Training Curve":
    st.title("Reconstructor Training Curve")
    hist = load_training_history()
    if hist is None:
        missing("results/training_history.json")
        st.stop()

    import plotly.graph_objects as go
    epochs = list(range(1, len(hist["train_loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=hist["train_loss"],
                             name="Train MSE", line=dict(color="#3a78c2")))
    fig.add_trace(go.Scatter(x=epochs, y=hist["val_loss"],
                             name="Val MSE", line=dict(color="#c23a3a")))
    best_ep = hist["best_epoch"]
    best_val = hist["best_val_loss"]
    fig.add_vline(x=best_ep, line_dash="dash", line_color="gray",
                  annotation_text=f"best epoch {best_ep} (val={best_val:.3f})")
    fig.update_layout(
        xaxis_title="Epoch", yaxis_title="MSE loss",
        title="MLP reconstructor — train / val MSE per epoch",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Best epoch", best_ep)
    col2.metric("Best val MSE", f"{best_val:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Plots
# ══════════════════════════════════════════════════════════════════════════
elif page == "Plots":
    st.title("Result Plots")

    plot_files = {
        "FVE histogram (per-sample, test split)": "fve_histogram.png",
        "Cosine similarity histogram (per-sample, test split)": "cosine_histogram.png",
        "Mean FVE per cluster": "fve_per_cluster.png",
        "Training loss curve": "training_loss.png",
    }
    any_found = False
    for title, fname in plot_files.items():
        p = RESULTS_DIR / fname
        if p.exists():
            st.subheader(title)
            st.image(str(p), use_container_width=True)
            any_found = True
        else:
            st.info(f"{fname} not found yet.")

    if not any_found:
        missing("result plots")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — Qualitative Examples
# ══════════════════════════════════════════════════════════════════════════
elif page == "Qualitative Examples":
    st.title("Qualitative Examples (best / median / worst)")
    df = load_qualitative()
    if df is None:
        missing("results/qualitative_examples.csv")
        st.stop()

    # Colour FVE column
    def colour_fve(val):
        try:
            v = float(val)
            if v >= 0.9:
                return "background-color: #d4edda"
            elif v >= 0.5:
                return "background-color: #fff3cd"
            else:
                return "background-color: #f8d7da"
        except Exception:
            return ""

    st.dataframe(
        df.style.map(colour_fve, subset=["fve"]),
        use_container_width=True,
        height=320,
    )

    st.divider()
    st.subheader("Inspect a single example")
    idx = st.selectbox("Select row", range(len(df)),
                       format_func=lambda i: f"Row {i} | cluster {df.iloc[i]['cluster']} | FVE {df.iloc[i]['fve']:.3f}")
    row = df.iloc[idx]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original text**")
        st.info(row.get("text", "—"))
        st.markdown("**Assigned explanation**")
        st.success(row.get("explanation", "—"))
    with col2:
        fve = float(row.get("fve", 0))
        cos = float(row.get("cosine", 0))
        mse = float(row.get("mse", 0))
        st.metric("FVE", f"{fve:.4f}",
                  delta="good" if fve > 0.8 else ("poor" if fve < 0.2 else "ok"),
                  delta_color="normal" if fve > 0.8 else "inverse")
        st.metric("Cosine similarity", f"{cos:.4f}")
        st.metric("MSE", f"{mse:.4f}")
        st.markdown(f"**Cluster:** {int(row['cluster'])}")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 6 — Cluster Explanations
# ══════════════════════════════════════════════════════════════════════════
elif page == "Cluster Explanations":
    st.title("Cluster Explanations")
    st.markdown(
        "Each cluster's natural-language explanation was generated by SmolLM2 "
        "prompted with the 5 representative texts closest to the cluster centroid."
    )

    clusters = load_clusters_llm()
    pc = load_per_cluster()

    if clusters is None:
        missing("data/clusters_llm.json / clusters.json")
        st.stop()

    # Merge cluster cards with per-cluster metrics if available
    cards = []
    for c in sorted(clusters, key=lambda x: x["cluster_id"]):
        row = {"cluster": c["cluster_id"], "size": c["size"],
               "explanation": c["explanation"]}
        if pc is not None:
            m = pc[pc["cluster"] == c["cluster_id"]]
            if not m.empty:
                row["fve_mean"] = round(float(m["fve_mean"].values[0]), 4)
                row["cosine_mean"] = round(float(m["cosine_mean"].values[0]), 4)
                row["n_test"] = int(m["n_test_samples"].values[0])
        cards.append(row)

    df_cards = pd.DataFrame(cards)

    # Sidebar filter
    sort_by = st.selectbox("Sort by", ["cluster", "fve_mean", "size"],
                           index=1 if "fve_mean" in df_cards.columns else 0)
    ascending = st.checkbox("Ascending", value=True)
    if sort_by in df_cards.columns:
        df_cards = df_cards.sort_values(sort_by, ascending=ascending)

    # Compact summary table
    display_cols = [c for c in ["cluster", "size", "n_test", "fve_mean", "cosine_mean"]
                    if c in df_cards.columns]
    st.dataframe(df_cards[display_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Expand a cluster")
    sel_cluster = st.selectbox(
        "Select cluster",
        df_cards["cluster"].tolist(),
        format_func=lambda c: (
            f"Cluster {c}"
            + (f" — FVE {df_cards[df_cards['cluster']==c]['fve_mean'].values[0]:.3f}" if "fve_mean" in df_cards.columns else "")
        ),
    )
    card = next(c for c in clusters if c["cluster_id"] == sel_cluster)

    st.markdown(f"**Explanation:** {card['explanation']}")
    st.markdown(f"**Cluster size:** {card['size']} samples")

    if card.get("representative_texts"):
        st.markdown("**Representative texts (closest to centroid):**")
        for i, t in enumerate(card["representative_texts"], 1):
            st.markdown(f"{i}. {t[:300]}{'…' if len(t)>300 else ''}")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 7 — Custom Text Demo  (honest, stable, interpretable)
# ══════════════════════════════════════════════════════════════════════════
elif page == "Custom Text Demo":
    st.title("Custom Text Demo")
    st.info(
        "**How this works:** Your text is passed through SmolLM2-360M to extract "
        "a 960-dimensional hidden-state vector. That vector is matched to the nearest "
        "of 32 pre-computed KMeans clusters. Each cluster has a precomputed English "
        "explanation. The explanation is then embedded and fed to the MLP reconstructor, "
        "which tries to recreate the original vector.\n\n"
        "⚠️ **Important:** The explanation shown is the *cluster's* explanation, not a "
        "generated description of your specific sentence. If the cluster assignment is "
        "uncertain or your text is unlike the training data, the explanation may be "
        "semantically unrelated to your input.",
        icon="ℹ️",
    )

    user_text = st.text_area(
        "Enter any text",
        value="The French Revolution began in 1789 and transformed European politics.",
        height=110,
    )

    run_btn = st.button("▶ Run pipeline", type="primary")

    if run_btn:
        # ── lazy imports & cached loaders ──────────────────────────────
        import torch
        from sklearn.cluster import KMeans
        from src.utils import get_device
        from src.model_utils import load_model
        from src.activations import extract_activations
        from src.reconstructor import MLPReconstructor

        @st.cache_resource
        def _load_model_cached():
            m = cfg["model"]
            device = get_device()
            return load_model(m["name"], m["layer_index"], m["dtype"], device), device

        @st.cache_resource
        def _load_kmeans_cached():
            acts_file = DATA_DIR / cfg["activations"]["file"]
            if not acts_file.exists():
                st.error(
                    f"**{acts_file.name}** not found — run scripts/02_collect_activations.py first.",
                    icon="🚫",
                )
                st.stop()
            all_acts = np.load(acts_file).astype(np.float32)
            km_path = DATA_DIR / "kmeans.joblib"
            if km_path.exists():
                import joblib
                km = joblib.load(km_path)
            else:
                # Fallback: refit with same seed (may produce different cluster
                # assignments than the pipeline run if KMeans converges differently)
                km = KMeans(
                    n_clusters=cfg["verbalizer"]["num_clusters"],
                    random_state=cfg["seed"],
                    n_init=cfg["verbalizer"]["kmeans_n_init"],
                )
                km.fit(all_acts)
            return km, all_acts

        @st.cache_resource
        def _load_reconstructor_cached():
            ckpt_path = CKPT_DIR / cfg["reconstructor"]["checkpoint_file"]
            if not ckpt_path.exists():
                return None
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            mdl = MLPReconstructor(
                input_dim=ckpt["input_dim"],
                output_dim=ckpt["output_dim"],
                hidden_dims=tuple(ckpt["hidden_dims"]),
                dropout=ckpt["dropout"],
            )
            mdl.load_state_dict(ckpt["state_dict"])
            mdl.eval()
            return mdl, ckpt["embedder_name"]

        with st.spinner("Loading model (first run: ~10 s, cached afterwards)…"):
            loaded, device     = _load_model_cached()
            km, all_acts       = _load_kmeans_cached()
            clusters_data      = load_clusters_llm() or []
            recon_result       = _load_reconstructor_cached()

        # ── extract activation ──────────────────────────────────────────
        with st.spinner("Extracting hidden-state activation…"):
            m   = cfg["model"]
            act = extract_activations(
                loaded, [user_text],
                max_length=m["max_length"],
                batch_size=1,
                pooling=m["pooling"],
            )                                    # shape (1, 960)
        act_vec = act[0].astype(np.float64)      # (960,)  — canonical dtype

        # ── top-3 cluster lookup ────────────────────────────────────────
        top3       = compute_top_k_clusters(act_vec, km, clusters_data, k=3)
        best       = top3[0]
        confidence = compute_assignment_confidence(top3)
        ood        = compute_ood_status(act_vec, all_acts, km, percentile=90.0)

        # ── reconstruct ─────────────────────────────────────────────────
        recon_vec  = None
        metrics    = None
        if recon_result is not None:
            recon_model, embedder_name = recon_result
            with st.spinner("Reconstructing activation from cluster explanation…"):
                from sentence_transformers import SentenceTransformer
                enc = SentenceTransformer(embedder_name, device=str(device))
                emb = enc.encode(
                    [best["explanation"]],
                    convert_to_numpy=True,
                ).astype(np.float32)             # (1, 384)
                with torch.no_grad():
                    y_hat = recon_model(
                        torch.from_numpy(emb)
                    ).numpy().ravel()            # (960,)
            recon_vec = y_hat.astype(np.float64)
            metrics   = safe_single_example_metrics(act_vec, recon_vec)

        # ════════════════════════════════════════════════════════════════
        # DISPLAY
        # ════════════════════════════════════════════════════════════════
        st.success("Done! Results below.")
        st.divider()

        # ── confidence & OOD banners ────────────────────────────────────
        conf_label = confidence["label"]
        conf_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(conf_label, "⚪")

        if ood["is_ood"]:
            st.warning(
                f"**Out-of-distribution warning.** Your input's distance to the nearest "
                f"cluster ({ood['custom_dist']:.1f}) is above the {90}th percentile of "
                f"training distances ({ood['threshold']:.1f}). "
                f"The assigned cluster explanation may be unreliable.",
                icon="⚠️",
            )

        if conf_label == "LOW":
            st.warning(
                f"**Low assignment confidence.** The nearest and second-nearest clusters "
                f"are almost equidistant (ratio = {confidence['ratio']:.3f}). "
                f"A small change in your text could flip the assignment.",
                icon="⚠️",
            )
        elif conf_label == "MEDIUM":
            st.info(
                f"**Medium assignment confidence** (ratio = {confidence['ratio']:.3f}). "
                f"The cluster assignment is plausible but not certain.",
                icon="ℹ️",
            )

        # ── nearest cluster explanation ─────────────────────────────────
        st.subheader("Nearest activation-cluster explanation")
        st.caption(
            "This is **not** a generated description of your exact sentence. "
            "It is the precomputed explanation for the nearest KMeans cluster, "
            "written by SmolLM2 based on that cluster's representative training texts."
        )
        st.info(best["explanation"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Nearest cluster", f"#{best['cluster_id']}")
        c2.metric("Cluster size", f"{best['size']} samples")
        c3.metric(
            f"{conf_color} Assignment confidence",
            conf_label,
            help="Based on ratio of distance-to-nearest vs distance-to-2nd-nearest centroid.",
        )

        # ── reconstruction metrics ──────────────────────────────────────
        st.divider()
        st.subheader("Reconstruction quality")
        st.caption(
            "The MLP takes the cluster explanation → sentence embedding → tries to "
            "recreate your 960-dimensional activation vector. These metrics compare "
            "the reconstruction to your original activation (not the cluster centroid)."
        )

        if metrics is not None:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(
                "Cosine similarity",
                f"{metrics['cosine']:.4f}",
                help="Direction match between original and reconstructed activation. "
                     "High cosine is expected — SmolLM2 activations all point in similar "
                     "directions regardless of content.",
            )
            mc2.metric(
                "MSE",
                f"{metrics['mse']:.3f}",
                help="Mean squared error across all 960 dimensions. Lower is better.",
            )
            mc3.metric(
                "RMSE",
                f"{metrics['rmse']:.3f}",
                help="Square root of MSE — same unit as the activation values.",
            )

            nr = metrics["norm_ratio"]
            st.metric(
                "Reconstruction norm ratio  (||recon|| / ||original||)",
                f"{nr:.3f}",
                help="Close to 1.0 means correct scale. Far from 1.0 means the "
                     "reconstruction has the right direction but wrong magnitude.",
            )

            # contextual warnings about cosine vs MSE
            if metrics["cosine"] > 0.98 and metrics["mse"] > 20:
                st.warning(
                    "Cosine is high but MSE is also high. This means the reconstruction "
                    "**points the same direction** as the original but has the **wrong scale**. "
                    "Cosine alone is misleading here.",
                    icon="⚠️",
                )
            if not (0.7 < nr < 1.4):
                st.warning(
                    f"Norm ratio = {nr:.2f} (expected ~1.0). "
                    "The reconstruction scale differs significantly from the original — "
                    "cosine similarity may look good while the actual vector is wrong.",
                    icon="⚠️",
                )
        else:
            st.info("Reconstructor checkpoint not found — run scripts 01–05 first.")

        # ── optional: generate direct explanation ───────────────────────
        st.divider()
        st.subheader("Direct text summary (optional)")
        st.caption(
            "Ask SmolLM2 to summarise your input directly in natural language. "
            "This is **separate** from the cluster-based explanation above."
        )
        if st.button("Generate direct summary of my input"):
            with st.spinner("Generating…"):
                import torch as _torch
                _tok  = loaded.tokenizer
                _mdl  = loaded.model
                prompt = (
                    f"Summarise the following text in one sentence:\n\n"
                    f"{user_text.strip()}\n\nSummary:"
                )
                enc_in = _tok(prompt, return_tensors="pt").to(loaded.device)
                with _torch.no_grad():
                    out_ids = _mdl.generate(
                        **enc_in,
                        max_new_tokens=60,
                        do_sample=False,
                        pad_token_id=_tok.pad_token_id,
                        eos_token_id=_tok.eos_token_id,
                    )
                full = _tok.decode(out_ids[0], skip_special_tokens=True)
                summary = full[len(prompt):].strip().split("\n")[0]
            st.success(f"**Direct text summary:** {summary}")
            st.caption("ℹ️ This is what SmolLM2 says about your specific sentence — "
                       "not the cluster explanation.")

        # ── top-3 cluster comparison ────────────────────────────────────
        st.divider()
        st.subheader("Top 3 nearest clusters")
        st.caption(
            "Showing the 3 closest clusters by Euclidean distance in activation space. "
            "A large gap between rank-1 and rank-2 distances means confident assignment."
        )
        for entry in top3:
            rank_icon = ["🥇", "🥈", "🥉"][entry["rank"] - 1]
            with st.expander(
                f"{rank_icon} Cluster #{entry['cluster_id']}  —  "
                f"distance {entry['distance']:.2f}  —  {entry['size']} samples",
                expanded=(entry["rank"] == 1),
            ):
                st.markdown(f"**Explanation:** {entry['explanation']}")
                if entry.get("representative_texts"):
                    st.markdown("**Representative training texts:**")
                    for i, t in enumerate(entry["representative_texts"][:3], 1):
                        st.markdown(f"{i}. {t[:240]}{'…' if len(t)>240 else ''}")

        # ── OOD details ─────────────────────────────────────────────────
        with st.expander("Assignment confidence & OOD details"):
            st.markdown(f"""
| Metric | Value |
|---|---|
| Distance to nearest centroid | `{ood['custom_dist']:.2f}` |
| Training median nearest-dist | `{ood['train_p50']:.2f}` |
| Training 90th-pct nearest-dist | `{ood['train_p90']:.2f}` |
| Percentile rank of this input | `{ood['percentile_rank']:.1f}th` |
| d₁ / d₂ ratio | `{confidence['ratio']:.4f}` |
| d₂ − d₁ margin | `{confidence['margin']:.2f}` |
| Confidence label | **{confidence['label']}** |
            """)
            st.caption(
                "Percentile rank: 90th means your input is farther from training clusters "
                "than 90% of the training data — likely out-of-distribution."
            )

        # ── raw activation debug ────────────────────────────────────────
        with st.expander("Debug: raw activation vector (first 20 of 960 dims)"):
            st.warning(
                "FVE is intentionally not shown for single custom inputs. "
                "FVE requires a reference variance computed from many samples. "
                "For one sample it produces astronomically large or negative values "
                "and is meaningless.",
                icon="ℹ️",
            )
            st.write("Original activation (first 20):", act_vec[:20].tolist())
            if recon_vec is not None:
                st.write("Reconstructed activation (first 20):", recon_vec[:20].tolist())

# ── footer ─────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption(
    "Core pipeline: run scripts 01–06\n"
    "then `streamlit run app/streamlit_app.py`"
)
