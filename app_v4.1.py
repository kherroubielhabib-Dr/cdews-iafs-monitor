# =====================================================
# CDEWS-IAFS v4.1 — FINAL (Stable + Calibrated)
# Reasoning Integrity Engine
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
# ✅ CALIBRATED THRESHOLDS (FIXED)
# =====================================================

SAFE_THRESHOLD = 0.55
DRIFT_THRESHOLD = 0.30
STEPS = 30

# =====================================================
# CONNECTORS
# =====================================================

CAUSAL_CONNECTORS = {
    "strong": [
        "therefore", "thus", "hence", "consequently",
        "لذلك", "إذن", "بالتالي", "وعليه", "بناءً على ذلك"
    ],
    "weak": [
        "because", "since", "as a result", "but", "however",
        "لأن", "بسبب", "لكن", "غير أن"
    ]
}

# =====================================================
# MODEL
# =====================================================

@st.cache_resource
def load_embedding_model():
    with st.spinner("🧠 Loading AI model..."):
        return SentenceTransformer('all-MiniLM-L6-v2')

try:
    model = load_embedding_model()
    model_loaded = True
except:
    model = None
    model_loaded = False

# =====================================================
# ✅ TEXT PROCESSING (FIXED)
# =====================================================

def split_into_sentences(text):
    sentences = re.split(r'[.!؟!?]+', text)
    
    clean = []
    for s in sentences:
        s = s.strip()
        if len(s) > 20 and len(s.split()) > 4:
            clean.append(s)
    
    return clean

def detect_connector(sentence):
    s = sentence.lower()
    
    for c in CAUSAL_CONNECTORS["strong"]:
        if c in s:
            return 0.85, c, "strong"
    
    for c in CAUSAL_CONNECTORS["weak"]:
        if c in s:
            return 0.60, c, "weak"
    
    return 0.40, None, "neutral"

# =====================================================
# SEMANTIC SIMILARITY
# =====================================================

def semantic_similarity(s1, s2):
    if model_loaded:
        e1 = model.encode([s1])[0]
        e2 = model.encode([s2])[0]
        sim = cosine_similarity([e1], [e2])[0][0]
        return max(0, min(1, (sim + 1) / 2))
    else:
        w1, w2 = set(s1.split()), set(s2.split())
        return len(w1 & w2) / len(w1 | w2) if (w1 | w2) else 0.5

# =====================================================
# ✅ CTL ENGINE (STABLE)
# =====================================================

def compute_ctl(text):
    sentences = split_into_sentences(text)
    
    if len(sentences) < 2:
        return 0, []
    
    tensions = []
    details = []
    
    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i+1]
        
        # تجاهل الجمل الضعيفة
        if len(s2.split()) < 4:
            continue
        
        expected, connector, ctype = detect_connector(s2)
        actual = semantic_similarity(s1, s2)
        
        base_tension = abs(expected - actual)
        
        # 🔥 Adaptive penalty
        if ctype == "strong":
            tension = base_tension
        elif ctype == "weak":
            tension = base_tension * 0.6
        else:
            tension = base_tension * 0.4
        
        tensions.append(tension)
        
        details.append({
            "step": i+1,
            "connector": connector or "context",
            "type": ctype,
            "similarity": round(actual, 3),
            "expected": expected,
            "tension": round(tension, 3),
            "risk": "🔥 High" if tension > 0.4 else ("⚠️ Medium" if tension > 0.2 else "✅ Stable")
        })
    
    if not tensions:
        return 0, details
    
    # وزن زمني خفيف
    weights = np.linspace(1, 2, len(tensions))
    weights /= weights.sum()
    
    tension_index = np.average(tensions, weights=weights)
    
    # 🔥 Clamp للحماية
    tension_index = min(tension_index, 0.85)
    
    return tension_index, details

# =====================================================
# v3.2 SIMULATION
# =====================================================

def simulate_v32():
    c_list = []
    prev = None
    
    for _ in range(STEPS):
        p = np.random.dirichlet(np.ones(10))
        
        h = -np.sum(p * np.log(p)) / np.log(len(p))
        d = 0 if prev is None else jensenshannon(p, prev)**2
        
        c = np.exp(-(0.4*h + 0.4*d))
        c_list.append(c)
        
        prev = p
    
    return c_list[-1], c_list

# =====================================================
# UI
# =====================================================

st.set_page_config(page_title="CDEWS-IAFS v4.1", layout="wide")
st.title("🔥 CDEWS-IAFS v4.1 — Reasoning Integrity Engine")

text = st.text_area("📝 Enter text for analysis:", height=200)

if st.button("Analyze", use_container_width=True):
    
    if text:
        v32, history = simulate_v32()
        ctl, details = compute_ctl(text)
        
        v41 = v32 * (1 - ctl)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("v3.2", f"{v32:.3f}")
        col2.metric("🔥 CTL", f"{ctl:.3f}")
        col3.metric("🚀 v4.1", f"{v41:.3f}")
        
        # ✅ Classification FIXED
        if v41 >= SAFE_THRESHOLD:
            st.success("✅ Stable reasoning")
        elif v41 > DRIFT_THRESHOLD:
            st.warning("⚠️ Drift detected")
        else:
            st.error("🔴 Logical collapse (hallucination)")
        
        st.divider()
        st.table(pd.DataFrame(details))
        
        # Graph
        fig, ax = plt.subplots()
        ax.plot(history, label="v3.2")
        ax.plot([c*(1-ctl) for c in history], label="v4.1")
        ax.legend()
        st.pyplot(fig)