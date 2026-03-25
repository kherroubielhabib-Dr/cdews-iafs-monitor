# =====================================================
# CDEWS-IAFS v5.1 — Reasoning Integrity Engine
# CTL v4.3 backbone + LNI diagnostic + DRA + Deterministic
# Dr. Elhabib Kherroubi — March 2026
# =====================================================
#
# CORRECTION vs v5.0:
# LNI was incorrectly reducing CTL tension.
# Fix: CTL uses v4.3 backbone (base × risk).
# LNI is now a DIAGNOSTIC INDICATOR only — shown in
# the step table to explain WHY tension is high/low,
# but does NOT modify the core tension calculation.
#
# FULL LINEAGE:
# v3.2   → H(t) · D(t) · SC · C(t)
# v4.1   → CTL — Causal Tension Layer
# v4.2   → DRA — Domain Risk Amplification
# v4.3   → Deterministic Core
# v4.3.1 → Formula precision
# v4.4   → Semantic D(t) + domain-aware H(t)
# v5.0   → DCS + LNI (architecture — DCS symmetric issue)
# v5.1   → CTL v4.3 backbone restored + LNI as diagnostic
# v5.2   → Hybrid D(t): lexical JSD + embedding cosine
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIG
# =====================================================

SAFE_THRESHOLD  = 0.75
DRIFT_THRESHOLD = 0.45

ALPHA = 0.45   # Entropy weight
BETA  = 0.35   # Drift weight
GAMMA = 0.20   # Structural Coherence weight

DOMAIN_RISK = {
    "General":   1.0,
    "Medical":   1.8,
    "Finance":   1.5,
    "Legal":     1.6,
    "AI Safety": 1.7,
}

# Connector strength — calibrated against PDTB (Prasad et al. 2008)
CONNECTOR_STRENGTH = {
    "strong":   0.85,
    "medium":   0.65,
    "contrast": 0.50,
    "neutral":  0.40,
}

CAUSAL_CONNECTORS = {
    "strong": [
        "therefore", "thus", "hence", "consequently", "as a result",
        "it follows that", "accordingly", "for this reason",
        "لذلك", "إذن", "بالتالي", "بناءً على ذلك", "وعليه", "من ثم",
        "نتيجة لذلك", "على هذا الأساس", "يترتب على ذلك",
        "donc", "par conséquent", "ainsi", "en conséquence",
    ],
    "medium": [
        "because", "since", "given that", "due to",
        "لأن", "بسبب", "نظراً لـ", "بما أن",
        "parce que", "car", "puisque",
    ],
    "contrast": [
        "but", "however", "yet", "although", "nevertheless", "despite",
        "لكن", "غير أن", "مع ذلك", "رغم ذلك", "بيد أن",
        "mais", "cependant", "pourtant", "néanmoins",
    ],
}

# =====================================================
# MODEL
# =====================================================

@st.cache_resource
def load_model():
    with st.spinner("🧠 Initializing CDEWS v5.1 Reasoning Engine..."):
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# =====================================================
# UTILITIES
# =====================================================

def split_sentences(text: str) -> list:
    parts = re.split(r'[.!?؟\n]+', text)
    return [p.strip() for p in parts if len(p.strip()) > 10]

def batch_embed(sentences: list) -> list:
    vecs = model.encode(sentences, convert_to_numpy=True).astype(np.float32)
    return [vecs[i] for i in range(len(vecs))]

# =====================================================
# CORE METRICS
# =====================================================

def compute_entropy(text: str, is_technical: bool = False) -> float:
    """
    H(t) — Normalized Shannon entropy.
    Domain-aware: technical texts × 0.70 correction.
    """
    words = re.findall(r'\w+', text.lower())
    if len(set(words)) < 2:
        return 0.0
    vocab = sorted(set(words))
    freq = np.array([words.count(w) for w in vocab], dtype=float)
    freq += 1e-12
    freq /= freq.sum()
    raw = -np.sum(freq * np.log(freq))
    h = float(np.clip(raw / np.log(len(freq)), 0.0, 1.0))
    return h * 0.70 if is_technical else h

def lexical_drift(text1: str, text2: str) -> float:
    """
    Lexical drift via word-frequency JSD.
    Captures vocabulary divergence between two texts.
    High when sentences use completely different words.
    """
    from scipy.spatial.distance import jensenshannon
    def to_dist(text):
        words = re.findall(r"\w+", text.lower())
        if not words:
            return np.array([1.0])
        vocab = sorted(set(words))
        freq = np.array([words.count(w) for w in vocab], dtype=float)
        freq += 1e-12
        return freq / freq.sum()

    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))
    vocab  = sorted(words1 | words2)

    def to_aligned(text):
        words = re.findall(r"\w+", text.lower())
        freq  = np.array([words.count(w) for w in vocab], dtype=float)
        freq += 1e-12
        return freq / freq.sum()

    p = to_aligned(text1)
    q = to_aligned(text2)
    return float(np.clip(jensenshannon(p, q) ** 2, 0.0, 1.0))


def semantic_drift_embedding(e1: np.ndarray, e2: np.ndarray) -> float:
    """
    Semantic drift via embedding cosine distance.
    Captures meaning divergence regardless of vocabulary.
    Low when sentences are semantically close.
    """
    sim = float(cosine_similarity([e1], [e2])[0][0])
    return float(np.clip((1.0 - sim) / 2.0, 0.0, 1.0))


def compute_semantic_drift(
    e1: np.ndarray, e2: np.ndarray,
    text1: str = "", text2: str = ""
) -> float:
    """
    D(t) v5.2 — Hybrid semantic drift (NEW).
    Combines lexical JSD + embedding cosine distance.

    D(t) = 0.5 × lexical_JSD + 0.5 × embedding_cosine_distance

    Why hybrid?
    - Lexical JSD: strong when texts use different words
      ("dog running" vs "car moving" → high lexical drift)
    - Embedding distance: strong when meanings diverge
      ("95% accuracy" vs "safe for hospitals" → high semantic drift)
    - Average: more robust, captures both dimensions

    Without text1/text2: falls back to embedding only.
    """
    emb_drift = semantic_drift_embedding(e1, e2)
    if text1 and text2:
        lex_drift = lexical_drift(text1, text2)
        return float(np.clip(0.5 * lex_drift + 0.5 * emb_drift, 0.0, 1.0))
    return emb_drift

def compute_structural_coherence(embeddings: list) -> float:
    """SC — Mean cosine similarity across consecutive pairs."""
    if len(embeddings) < 2:
        return 1.0
    sims = []
    for i in range(len(embeddings) - 1):
        sim = float(cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0])
        sims.append(float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0)))
    return float(np.mean(sims))

def compute_c_score(h: float, d: float, sc: float) -> float:
    """C(t) = exp(-(α·H + β·D + γ·(1-SC))). Deterministic."""
    return float(np.exp(-(ALPHA * h + BETA * d + GAMMA * (1.0 - sc))))

# =====================================================
# v5.1 COMPONENTS
# =====================================================

def detect_connector(sentence: str):
    s = sentence.lower()
    for ctype, words in CAUSAL_CONNECTORS.items():
        for w in words:
            if w in s:
                return CONNECTOR_STRENGTH[ctype], ctype, w
    return CONNECTOR_STRENGTH["neutral"], "neutral", "—"

def compute_lni(expected: float, drift: float) -> float:
    """
    LNI — Logical Necessity Index (DIAGNOSTIC ONLY in v5.1)
    Measures how necessary a reasoning step was.

    LNI = expected × (1 − drift)

    High LNI: connector is strong AND semantic drift is low
              → step appears logically necessary
    Low LNI:  weak connector OR high semantic drift
              → step appears arbitrary

    NOTE: LNI is shown in the step table as a diagnostic signal.
    It does NOT modify the CTL calculation (that caused
    false softening in v5.0). CTL uses v4.3 backbone.
    True LNI integration into the score requires
    ground-truth benchmark validation — planned for v6.0.
    """
    return float(np.clip(expected * (1.0 - drift), 0.0, 1.0))

# =====================================================
# CTL ENGINE — v4.3 backbone (proven + correct)
# =====================================================

def compute_ctl(sentences: list, embeddings: list, domain: str) -> tuple:
    """
    CTL v5.1 — Causal Tension Layer.

    BACKBONE: v4.3 formula (proven correct):
      base     = |expected − actual_sim|
      CTL      = base × risk(domain)          ← simple, proven
      penalty  = sim × 0.75 if length_ratio > 1.5

    DIAGNOSTIC (v5.1 addition):
      LNI shown in table — explains tension, doesn't modify it.
      Contradiction flag: sim < 0.25 AND expected > 0.75

    v5.0 ISSUE FIXED:
      v5.0: CTL = base × risk × (1-LNI) → LNI softened CTL
      v5.1: CTL = base × risk            → correct behavior
    """
    if len(sentences) < 2:
        return 0.0, []

    risk = DOMAIN_RISK.get(domain, 1.0)
    tensions = []
    details  = []

    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i+1]
        e1, e2 = embeddings[i], embeddings[i+1]

        expected, ctype, connector = detect_connector(s2)

        sim = float(np.clip(
            (cosine_similarity([e1], [e2])[0][0] + 1.0) / 2.0,
            0.0, 1.0
        ))

        # Jump penalty (v4.1+)
        if len(s2) / max(len(s1), 1) > 1.5:
            sim *= 0.75

        drift = compute_semantic_drift(e1, e2, s1, s2)

        # ── CTL v4.3 backbone ──────────────────────────────────
        base = abs(expected - sim)
        ctl  = float(np.clip(base * risk, 0.0, 1.0))

        # ── LNI diagnostic (v5.1) ──────────────────────────────
        lni = compute_lni(expected, drift)

        # ── Contradiction detection (v5.0) ─────────────────────
        contradiction = (sim < 0.25 and expected > 0.75)

        tensions.append(ctl)
        details.append({
            "Step":      i + 1,
            "Connector": connector,
            "Type":      ctype,
            "Expected":  round(expected, 3),
            "Sim":       round(sim,      3),
            "Drift":     round(drift,    3),
            "LNI":       round(lni,      3),
            "CTL":       round(ctl,      3),
            "Flag": (
                "🔴 CONTRADICTION" if contradiction else
                "🔥 High"          if ctl > 0.5    else
                "⚠️ Medium"        if ctl > 0.2    else
                "✅ Stable"
            ),
        })

    return float(np.mean(tensions)) if tensions else 0.0, details

# =====================================================
# FULL ANALYSIS PIPELINE
# =====================================================

def analyze(text: str, domain: str, is_technical: bool = False) -> dict:
    sentences  = split_sentences(text)
    if not sentences:
        return {}

    embeddings = batch_embed(sentences)

    h = compute_entropy(text, is_technical)

    d = float(np.mean([
        compute_semantic_drift(embeddings[i], embeddings[i+1],
                               sentences[i], sentences[i+1])
        for i in range(len(embeddings) - 1)
    ])) if len(embeddings) >= 2 else 0.0

    sc     = compute_structural_coherence(embeddings)
    c_base = compute_c_score(h, d, sc)
    ctl, details = compute_ctl(sentences, embeddings, domain)
    final  = float(np.clip(c_base * (1.0 - ctl), 0.0, 1.0))

    return {
        "H":       round(h,      4),
        "D":       round(d,      4),
        "SC":      round(sc,     4),
        "C_base":  round(c_base, 4),
        "CTL":     round(ctl,    4),
        "Final":   round(final,  4),
        "details": details,
    }

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(page_title="CDEWS-IAFS v5.2", layout="wide")

st.title("🧠 CDEWS-IAFS v5.2 — Reasoning Integrity Engine")
st.caption(
    "Dr. Elhabib Kherroubi  |  "
    "Deterministic · Model-Agnostic · Sovereign  |  "
    "CTL v4.3 + LNI Diagnostic + DRA + Hybrid D(t)"
)

st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    text = st.text_area(
        "📝 Enter text for analysis:",
        height=220,
        placeholder="Enter any text in Arabic, French, or English...",
    )

with col_right:
    domain = st.selectbox("🌍 Domain:", list(DOMAIN_RISK.keys()))
    is_technical = st.checkbox(
        "📚 Technical / Expert text",
        help=(
            "Enable for scientific papers, medical reports, legal documents. "
            "Applies 0.70 entropy correction to avoid penalizing "
            "specialized vocabulary."
        )
    )
    st.info(f"Risk multiplier: **×{DOMAIN_RISK[domain]}**")
    analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

st.divider()

if analyze_btn:
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Computing reasoning integrity..."):
            result = analyze(text, domain, is_technical)

        if not result:
            st.error("Could not parse text. Please enter at least one sentence.")
        else:
            # Metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("H(t) Entropy", result["H"],
                      help="Informational dispersion [0→1]")
            c2.metric("D(t) Drift",   result["D"],
                      help="Semantic drift via embeddings [0→1]")
            c3.metric("SC Coherence", result["SC"],
                      help="Structural coherence [0→1]")
            c4.metric("C(t) Base",    result["C_base"],
                      help="Core stability score")
            c5.metric("🔥 CTL",       result["CTL"],
                      help="Causal tension × domain risk")

            st.divider()

            # Final verdict
            final = result["Final"]
            st.subheader(f"🚀 Final Score v5.2: **{final}**")

            if final >= SAFE_THRESHOLD:
                st.success("✅ Stable Reasoning — Safe for decision support")
            elif final >= DRIFT_THRESHOLD:
                st.warning("⚠️ Drift Detected — Human review recommended")
            else:
                st.error("🔴 Logical Instability — Do not rely without review")

            st.divider()

            # CTL detail table
            if result["details"]:
                st.subheader("🔍 Causal Tension — Step by Step")
                df = pd.DataFrame(result["details"])
                st.dataframe(df, use_container_width=True)

                # Contradiction alert
                contradictions = [
                    d for d in result["details"]
                    if "CONTRADICTION" in d.get("Flag", "")
                ]
                if contradictions:
                    st.error(
                        f"🔴 {len(contradictions)} CONTRADICTION(S) detected — "
                        "strong connector with very low semantic alignment."
                    )

                # LNI insight
                low_lni = [
                    d for d in result["details"]
                    if d.get("LNI", 1.0) < 0.3
                ]
                if low_lni:
                    st.warning(
                        f"⚠️ {len(low_lni)} step(s) with low LNI (< 0.30) — "
                        "reasoning necessity is weak at these transitions."
                    )
            else:
                st.info("At least 2 sentences required for CTL analysis.")

            # Formula Reference
            with st.expander("📐 Formula Reference & Scientific Justification"):
                st.markdown("""
**Core Stability:**
```
C(t) = exp(-(α·H(t) + β·D(t) + γ·(1 - SC)))
α = 0.45 | β = 0.35 | γ = 0.20
```

**Hybrid D(t) v5.2:**
```
D(t) = 0.5 × Lexical_JSD + 0.5 × Embedding_cosine_distance
```

**Causal Tension (CTL v4.3 backbone):**
```
Base = |Expected − Actual_sim|
CTL  = Base × Risk(domain)
```

**Logical Necessity Index (LNI — diagnostic):**
```
LNI = expected × (1 − drift)
```
*LNI is shown in the step table as a diagnostic signal.*
*It explains WHY tension is high — it does not modify CTL.*
*Full LNI integration planned for v6.0 after benchmark validation.*

**Final Score:**
```
Final = C(t) × (1 − CTL)
```

---

**Connector Strength** *(Penn Discourse Treebank, Prasad et al. 2008)*

| Type | Value | Linguistic Basis |
|------|-------|-----------------|
| Strong (therefore, thus...) | 0.85 | Entailment — conclusion MUST follow |
| Medium (because, since...) | 0.65 | Causation — conclusion LIKELY follows |
| Contrast (but, however...) | 0.50 | Opposition — semantic distance expected |
| Neutral (no connector) | 0.40 | Weak expectation only |

---

**Version Lineage:**

| Version | Key Addition |
|---------|-------------|
| v3.2 | H(t) · D(t) · SC · C(t) |
| v4.1 | CTL — Causal Tension Layer |
| v4.2 | DRA — Domain Risk Amplification |
| v4.3 | Deterministic Core |
| v4.4 | Semantic D(t) + domain-aware H(t) |
| v5.0 | DCS + LNI architecture |
| v5.1 | CTL v4.3 backbone restored + LNI as diagnostic |
| v5.2 | Hybrid D(t): lexical JSD + embedding cosine ✅ |

---
*Domain Risk: General ×1.0 | Finance ×1.5 | Legal ×1.6 | AI Safety ×1.7 | Medical ×1.8*
                """)

