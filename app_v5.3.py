# =====================================================
# CDEWS-IAFS v5.3 --- Reasoning Integrity Engine
# Dr. Elhabib Kherroubi --- March 2026
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

SAFE_THRESHOLD = 0.75
DRIFT_THRESHOLD = 0.45

ALPHA = 0.45
BETA = 0.35
GAMMA = 0.20

DOMAIN_RISK = {
    "General": 1.0,
    "Medical": 1.8,
    "Finance": 1.5,
    "Legal": 1.6,
    "AI Safety": 1.7,
}

CONNECTOR_STRENGTH = {
    "strong": 0.85,
    "medium": 0.65,
    "contrast": 0.50,
    "neutral": 0.40,
}

# =====================================================
# CAUSAL CONNECTORS (مع الروابط العربية القوية)
# =====================================================

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
    with st.spinner("Loading model..."):
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
    from scipy.spatial.distance import jensenshannon
    
    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))
    vocab = sorted(words1 | words2)
    
    def to_aligned(text):
        words = re.findall(r"\w+", text.lower())
        freq = np.array([words.count(w) for w in vocab], dtype=float)
        freq += 1e-12
        return freq / freq.sum()
    
    p = to_aligned(text1)
    q = to_aligned(text2)
    return float(np.clip(jensenshannon(p, q) ** 2, 0.0, 1.0))

def semantic_drift_embedding(e1: np.ndarray, e2: np.ndarray) -> float:
    sim = float(cosine_similarity([e1], [e2])[0][0])
    return float(np.clip((1.0 - sim) / 2.0, 0.0, 1.0))

def compute_semantic_drift(e1: np.ndarray, e2: np.ndarray, text1: str = "", text2: str = "") -> float:
    emb_drift = semantic_drift_embedding(e1, e2)
    if text1 and text2:
        lex_drift = lexical_drift(text1, text2)
        return float(np.clip(0.5 * lex_drift + 0.5 * emb_drift, 0.0, 1.0))
    return emb_drift

def compute_structural_coherence(embeddings: list) -> float:
    if len(embeddings) < 2:
        return 1.0
    
    sims = []
    for i in range(len(embeddings) - 1):
        sim = float(cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0])
        sims.append(float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0)))
    
    return float(np.mean(sims))

def compute_c_score(h: float, d: float, sc: float) -> float:
    return float(np.exp(-(ALPHA * h + BETA * d + GAMMA * (1.0 - sc))))

# =====================================================
# ENHANCED COMPONENTS
# =====================================================

def detect_connector(sentence: str):
    s = sentence.lower()
    for ctype, words in CAUSAL_CONNECTORS.items():
        for w in words:
            if w in s:
                return CONNECTOR_STRENGTH[ctype], ctype, w
    return CONNECTOR_STRENGTH["neutral"], "neutral", "---"

def compute_lni(expected: float, drift: float) -> float:
    return float(np.clip(expected * (1.0 - drift), 0.0, 1.0))

def compute_lga(expected: float, sim: float, drift: float, domain: str) -> tuple:
    is_logic_gap = False
    gap_type = None
    
    if drift >= 0.40 and expected >= 0.80:
        is_logic_gap = True
        gap_type = "Semantic Gap"
    elif sim < 0.65 and expected >= 0.80:
        is_logic_gap = True
        gap_type = "Weak Support"
    
    if is_logic_gap:
        factor = 1.5 if domain in ["Medical", "Finance", "AI Safety"] else 1.25
        return factor, True, gap_type
    
    return 1.0, False, None

# =====================================================
# CTL ENGINE
# =====================================================

def compute_ctl(sentences: list, embeddings: list, domain: str) -> tuple:
    if len(sentences) < 2:
        return 0.0, [], []
    
    risk = DOMAIN_RISK.get(domain, 1.0)
    tensions = []
    details = []
    all_drifts = []
    
    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i+1]
        e1, e2 = embeddings[i], embeddings[i+1]
        
        expected, ctype, connector = detect_connector(s2)
        
        sim = float(np.clip(
            (cosine_similarity([e1], [e2])[0][0] + 1.0) / 2.0,
            0.0, 1.0
        ))
        
        if len(s2) / max(len(s1), 1) > 1.5:
            sim *= 0.75
        
        drift = compute_semantic_drift(e1, e2, s1, s2)
        all_drifts.append(drift)
        
        base = abs(expected - sim)
        base_ctl = float(np.clip(base * risk, 0.0, 1.0))
        
        lga_factor, is_gap, gap_type = compute_lga(expected, sim, drift, domain)
        final_ctl = float(np.clip(base_ctl * lga_factor, 0.0, 1.0))
        
        lni = compute_lni(expected, drift)
        contradiction = (sim < 0.25 and expected > 0.75)
        
        if is_gap:
            flag = f"💣 {gap_type}"
        elif contradiction:
            flag = "🔴 CONTRADICTION"
        elif final_ctl > 0.5:
            flag = "🔥 High Tension"
        elif final_ctl > 0.2:
            flag = "⚠️ Medium Tension"
        else:
            flag = "✅ Stable"
        
        tensions.append(final_ctl)
        details.append({
            "Step": i + 1,
            "Connector": connector,
            "Type": ctype,
            "Expected": round(expected, 3),
            "Sim": round(sim, 3),
            "Drift": round(drift, 3),
            "LNI": round(lni, 3),
            "LGA": round(lga_factor, 2),
            "CTL": round(final_ctl, 3),
            "Flag": flag,
        })
    
    return float(np.mean(tensions)) if tensions else 0.0, details, all_drifts

# =====================================================
# FULL ANALYSIS
# =====================================================

def analyze(text: str, domain: str, is_technical: bool = False) -> dict:
    sentences = split_sentences(text)
    if not sentences:
        return {}
    
    embeddings = batch_embed(sentences)
    
    h = compute_entropy(text, is_technical)
    sc = compute_structural_coherence(embeddings)
    
    ctl, details, all_drifts = compute_ctl(sentences, embeddings, domain)
    
    d = float(np.mean(all_drifts)) if all_drifts else 0.0
    
    c_base = compute_c_score(h, d, sc)
    final = float(np.clip(c_base * (1.0 - ctl), 0.0, 1.0))
    
    return {
        "H": round(h, 4),
        "D": round(d, 4),
        "SC": round(sc, 4),
        "C_base": round(c_base, 4),
        "CTL": round(ctl, 4),
        "Final": round(final, 4),
        "details": details,
    }

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(page_title="CDEWS-IAFS v5.3", layout="wide")

st.title("CDEWS-IAFS v5.3 --- Reasoning Integrity Engine")
st.caption("Dr. Elhabib Kherroubi | Deterministic · Model-Agnostic · Sovereign")

st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    text = st.text_area(
        "Enter text for analysis:",
        height=220,
        placeholder="Enter any text in Arabic, French, or English...",
    )

with col_right:
    domain = st.selectbox("Domain:", list(DOMAIN_RISK.keys()))
    is_technical = st.checkbox("Technical / Expert text")
    st.info(f"Risk multiplier: x{DOMAIN_RISK[domain]}")

analyze_btn = st.button("Analyze", use_container_width=True, type="primary")

st.divider()

if analyze_btn:
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Computing..."):
            result = analyze(text, domain, is_technical)
        
        if not result:
            st.error("Could not parse text. Please enter at least one sentence.")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("H(t) Entropy", result["H"])
            c2.metric("D(t) Drift", result["D"])
            c3.metric("SC Coherence", result["SC"])
            c4.metric("C(t) Base", result["C_base"])
            c5.metric("CTL", result["CTL"])
            
            st.divider()
            
            final = result["Final"]
            st.subheader(f"Final Score: {final}")
            
            if final >= SAFE_THRESHOLD:
                st.success("Stable Reasoning")
            elif final >= DRIFT_THRESHOLD:
                st.warning("Drift Detected")
            else:
                st.error("Logical Instability")
            
            st.divider()
            
            if result["details"]:
                st.subheader("Causal Tension Details")
                df = pd.DataFrame(result["details"])
                
                def color_flags(val):
                    if "💣" in str(val):
                        return 'background-color: #ff6b6b; color: white'
                    elif "🔴" in str(val):
                        return 'background-color: #ff4444; color: white'
                    elif "🔥" in str(val):
                        return 'background-color: #ffa500'
                    elif "⚠️" in str(val):
                        return 'background-color: #ffd700'
                    return ''
                
                styled_df = df.style.applymap(color_flags, subset=['Flag'])
                st.dataframe(styled_df, use_container_width=True)
                
                logic_gaps = [d for d in result["details"] if "💣" in d.get("Flag", "")]
                low_lni = [d for d in result["details"] if d.get("LNI", 1.0) < 0.3]
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if logic_gaps:
                        st.error(f"Logic Gaps: {len(logic_gaps)}")
                    else:
                        st.success("No logic gaps detected")
                with col_b:
                    if low_lni:
                        st.warning(f"Low LNI steps: {len(low_lni)}")
                    else:
                        st.success("Logical necessity maintained")
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Export CSV", csv, "cdews_results.csv", "text/csv")
            else:
                st.info("At least 2 sentences required.")
