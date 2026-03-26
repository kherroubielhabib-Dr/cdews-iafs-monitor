# =====================================================
# CDEWS-IAFS v4.1 — FINAL CORE
# Causal Tension Layer + Reasoning Validation
# Dr. Elhabib Kherroubi
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import warnings

warnings.filterwarnings('ignore')

# =====================================================
# CONFIG
# =====================================================

SAFE_THRESHOLD = 0.75
DRIFT_THRESHOLD = 0.4
STEPS = 30

# =====================================================
# CONNECTORS
# =====================================================

CAUSAL_CONNECTORS = {
    "strong": [
        "therefore", "thus", "so", "hence", "consequently", "accordingly",
        "لذلك", "إذن", "بالتالي", "وعليه", "من ثم", "بناءً على ذلك"
    ],
    "weak": [
        "because", "since", "as a result", "however", "but", "yet",
        "لأن", "بسبب", "لكن", "غير أن", "مع ذلك"
    ]
}

# =====================================================
# MODEL LOAD
# =====================================================

@st.cache_resource
def load_model():
    with st.spinner("🧠 Initializing Sovereign Intelligence..."):
        return SentenceTransformer('all-MiniLM-L6-v2')

try:
    embed_model = load_model()
    model_loaded = True
except:
    embed_model = None
    model_loaded = False

# =====================================================
# CORE FUNCTIONS
# =====================================================

def split_into_sentences(text):
    sentences = re.split(r'[.!؟!?\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

# -------------------------------
# CLAIM EXTRACTION (🔥 NEW)
# -------------------------------

def extract_claim_structure(sentence):
    sentence = sentence.lower()

    return {
        "has_quantifier": bool(re.search(r'\d+%|\d+', sentence)),
        "has_superlative": any(w in sentence for w in [
            "best", "all", "always", "never",
            "أفضل", "كل", "دائما"
        ]),
        "scope": "global" if any(w in sentence for w in [
            "all", "every", "جميع", "كل"
        ]) else "local"
    }

# -------------------------------
# CAUSAL GAP (🔥 CORE INNOVATION)
# -------------------------------

def causal_gap_score(c1, c2):
    gap = 0

    # Local → Global jump
    if c1["scope"] == "local" and c2["scope"] == "global":
        gap += 0.4

    # Strong generalization
    if c2["has_superlative"]:
        gap += 0.3

    # Weak evidence → strong claim
    if c1["has_quantifier"] and c2["has_superlative"]:
        gap += 0.3

    return min(1.0, gap)

# -------------------------------
# CONFIDENCE PENALTY
# -------------------------------

def confidence_penalty(sentence):
    words = [
        "definitely", "always", "never", "certainly", "best",
        "أفضل", "دائما", "مؤكد"
    ]
    score = sum([1 for w in words if w in sentence.lower()])
    return min(0.3, score * 0.1)

# -------------------------------
# CONNECTOR DETECTION
# -------------------------------

def detect_connector(sentence):
    s = sentence.lower()

    for c in CAUSAL_CONNECTORS["strong"]:
        if c in s:
            return 0.9, "strong", c

    for c in CAUSAL_CONNECTORS["weak"]:
        if c in s:
            return 0.6, "weak", c

    return 0.3, "none", None

# -------------------------------
# SEMANTIC SIMILARITY
# -------------------------------

def semantic_similarity(s1, s2):
    if model_loaded and embed_model:
        e1 = embed_model.encode([s1])[0]
        e2 = embed_model.encode([s2])[0]
        sim = cosine_similarity([e1], [e2])[0][0]
        return max(0, min(1, (sim + 1) / 2))
    else:
        w1, w2 = set(s1.split()), set(s2.split())
        return len(w1 & w2) / len(w1 | w2) if (w1 | w2) else 0.5

# =====================================================
# CTL ENGINE
# =====================================================

def compute_ctl(text):
    sentences = split_into_sentences(text)

    if len(sentences) < 2:
        return 0, []

    tensions = []
    details = []

    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i+1]

        expected, level, connector = detect_connector(s2)
        actual = semantic_similarity(s1, s2)

        # CLAIM ANALYSIS
        c1 = extract_claim_structure(s1)
        c2 = extract_claim_structure(s2)

        gap = causal_gap_score(c1, c2)
        penalty = confidence_penalty(s2)

        # FINAL TENSION
        tension = abs(expected - actual)
        tension += gap
        tension += penalty

        if level == "strong":
            tension *= 1.1

        tension = min(1.0, tension)

        tensions.append(tension)

        details.append({
            "Step": i+1,
            "Connector": connector or "Free",
            "Similarity": round(actual, 3),
            "Gap": round(gap, 3),
            "Penalty": round(penalty, 3),
            "Tension": round(tension, 3),
            "Risk": "🔥 High" if tension > 0.6 else ("⚠️ Medium" if tension > 0.3 else "✅ Stable")
        })

    weights = np.exp(np.linspace(0, 1, len(tensions)))
    weights /= weights.sum()

    index = np.average(tensions, weights=weights)

    return index, details

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(page_title="CDEWS-IAFS v4.1 FINAL", layout="wide")
st.title("🔥 CDEWS-IAFS v4.1 — Sovereign Reasoning Engine")

text = st.text_area("📝 أدخل النص:", height=200)

if st.button("🚀 Analyze Reasoning"):

    if text:

        # v3.2 simulation
        c_list = []
        prev = None

        for _ in range(STEPS):
            p = np.random.dirichlet(np.ones(10))
            h = -np.sum(p * np.log(p))
            d = 0 if prev is None else jensenshannon(p, prev)**2
            c = np.exp(-(0.45*h + 0.35*d))
            c_list.append(c)
            prev = p

        base = c_list[-1]

        # v4.1 CTL
        ctl_index, details = compute_ctl(text)

        final = base * (1 - ctl_index)

        col1, col2, col3 = st.columns(3)
        col1.metric("v3.2", f"{base:.3f}")
        col2.metric("🔥 CTL", f"{ctl_index:.3f}")
        col3.metric("🚀 Final v4.1", f"{final:.3f}")

        if final >= SAFE_THRESHOLD:
            st.success("✅ Reasoning is logically valid")
        elif final > DRIFT_THRESHOLD:
            st.warning("⚠️ Reasoning drift detected")
        else:
            st.error("🔴 Logical collapse (hallucination)")

        st.divider()
        st.subheader("🔍 Detailed Analysis")
        st.table(pd.DataFrame(details))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(c_list, label="v3.2")
        ax.plot([c*(1-ctl_index) for c in c_list], label="v4.1")
        ax.legend()
        st.pyplot(fig)

    else:
        st.warning("Enter text first")

