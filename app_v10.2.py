# =====================================================
# CDEWS-IAFS v10.2 — النسخة الخرافية الكاملة
# Streamlit Application
# Dr. Elhabib Kherroubi — March 2026
# =====================================================

import streamlit as st
import numpy as np
import re
import math
from collections import Counter

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(page_title="CDEWS-IAFS v10.2", layout="wide", page_icon="👑")

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

# =====================================================
# CAUSAL SAFE PATTERNS
# =====================================================

CAUSAL_SAFE = [
    ("appendicitis", "appendectomy"), ("appendicitis", "surgery"),
    ("inflammation", "surgery"), ("inflammation", "treatment"),
    ("infection", "treatment"), ("fracture", "immobilization"),
    ("pneumonia", "antibiotics"), ("hypertension", "medication"),
    ("diabetes", "insulin"), ("التهاب", "استئصال"), ("التهاب", "علاج"),
    ("كسر", "تثبيت"), ("زائدة", "استئصال"), ("fever", "antipyretic")
]

# =====================================================
# SEVERITY PATTERNS
# =====================================================

SEVERE_WORDS = ["surgery", "lobectomy", "amputation", "chemotherapy", 
                "استئصال", "قاتل", "جراحة", "إزالة", "death", "kill", "موت"]
MILD_WORDS = ["slight", "minor", "small", "minimal", 
              "طفيف", "بسيط", "قليل", "خفيف", "محتمل"]

# =====================================================
# STRONG CONNECTORS
# =====================================================

STRONG_CONNECTORS = ["therefore", "thus", "hence", "consequently", 
                     "as a result", "it follows that", 
                     "لذلك", "إذن", "بالتالي", "بناءً على ذلك", "وعليه"]

# =====================================================
# UTILITIES
# =====================================================

def tokenize(text):
    return re.findall(r'[\u0600-\u06FF\w]+', text.lower())

def compute_entropy(text):
    words = tokenize(text)
    if len(words) < 2:
        return 0.0
    freq = Counter(words)
    total = len(words)
    h = 0.0
    for count in freq.values():
        p = count / total
        h -= p * math.log(p)
    max_h = math.log(len(freq))
    return min(1.0, h / max_h) if max_h > 0 else 0.0

def split_sentences(text):
    sentences = re.split(r'[.!?؟]\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 8]

# =====================================================
# HYBRID DRIFT
# =====================================================

def lexical_drift(t1, t2):
    w1 = tokenize(t1)
    w2 = tokenize(t2)
    vocab = list(set(w1 + w2))
    
    def distribution(words):
        freq = np.array([words.count(v) for v in vocab], dtype=float)
        freq += 1e-12
        return freq / freq.sum()
    
    p = distribution(w1)
    q = distribution(w2)
    m = (p + q) / 2
    
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    js = (kl_pm + kl_qm) / 2
    
    return min(1.0, max(0.0, js))

def embedding_drift(t1, t2, dim=96):
    def hash_embed(text):
        vec = np.zeros(dim)
        words = tokenize(text)
        for idx, w in enumerate(words):
            h = 0
            for c in w:
                h = (h * 31 + ord(c)) & 0xffffffff
            vec[abs(h) % dim] += 1.0 / (idx + 1)
        norm = np.sqrt(np.sum(vec ** 2)) or 1.0
        return vec / norm
    
    v1 = hash_embed(t1)
    v2 = hash_embed(t2)
    sim = np.dot(v1, v2)
    sim = max(0.0, min(1.0, (sim + 1.0) / 2.0))
    return (1.0 - sim) / 2.0, sim

def hybrid_drift(t1, t2):
    lex = lexical_drift(t1, t2)
    emb_drift, emb_sim = embedding_drift(t1, t2)
    drift = 0.5 * lex + 0.5 * emb_drift
    return drift, emb_sim

# =====================================================
# DETECT CONNECTOR
# =====================================================

def detect_connector(sentence):
    s = sentence.lower()
    for w in STRONG_CONNECTORS:
        if re.search(r'\b' + re.escape(w) + r'\b', s):
            return 0.85, "strong"
    return 0.40, "neutral"

# =====================================================
# CAUSAL SAFE ADJUSTMENT
# =====================================================

def causal_safe_adjust(t1, t2, drift, sim):
    s1 = t1.lower()
    s2 = t2.lower()
    for a, b in CAUSAL_SAFE:
        if a in s1 and b in s2:
            return drift * 0.6, min(0.95, sim + 0.12), f"{a} → {b}"
    return drift, sim, None

# =====================================================
# SEVERITY MISMATCH
# =====================================================

def detect_severity_mismatch(t1, t2):
    s1 = t1.lower()
    s2 = t2.lower()
    has_mild = any(w in s1 for w in MILD_WORDS)
    has_severe = any(w in s2 for w in SEVERE_WORDS)
    if has_mild and has_severe:
        return True, 0.6, "⚠ Severity mismatch: mild premise → severe conclusion"
    return False, 0.0, None

# =====================================================
# MAIN ANALYSIS ENGINE
# =====================================================

def analyze_text_engine(text, domain):
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return None
    
    risk = DOMAIN_RISK.get(domain, 1.0)
    drift_thr = 0.45 if risk == 1.0 else 0.50
    
    H = compute_entropy(text)
    n = len(sentences)
    
    drifts = []
    sims = []
    steps = []
    
    for i in range(n - 1):
        s1, s2 = sentences[i], sentences[i+1]
        
        # Hybrid drift
        drift, sim = hybrid_drift(s1, s2)
        
        # Severity detection
        severity_mismatch, severity_gap, severity_reason = detect_severity_mismatch(s1, s2)
        
        # Causal safe adjustment
        drift, sim, pattern = causal_safe_adjust(s1, s2, drift, sim)
        
        # Connector detection
        expected, conn_type = detect_connector(s2)
        expected = max(0.30, min(0.95, expected - drift * 0.25 + sim * 0.12))
        
        # Base tension
        base = abs(expected - sim)
        
        # LGA v2 amplification
        factor = 1.0
        if severity_mismatch:
            factor = 1.8
        elif drift >= drift_thr and expected >= 0.80:
            factor = 1.5
        
        ctl = min(1.0, base * risk * factor)
        
        drifts.append(drift)
        sims.append(sim)
        
        # Determine tag
        if pattern:
            tag = "safe"
        elif severity_mismatch:
            tag = "gap"
        elif ctl > 0.45:
            tag = "drift"
        else:
            tag = "neutral"
        
        steps.append({
            "text": s2[:80] + ("..." if len(s2) > 80 else ""),
            "full_text": s2,
            "drift": drift,
            "sim": sim,
            "expected": expected,
            "ctl": ctl,
            "tag": tag,
            "pattern": pattern,
            "severity_reason": severity_reason
        })
    
    D = np.mean(drifts)
    SC = np.mean(sims)
    C_base = math.exp(-(ALPHA * H + BETA * D + GAMMA * (1 - SC)))
    
    tensions = [s["ctl"] for s in steps]
    max_t = max(tensions) if tensions else 0
    mean_t = np.mean(tensions) if tensions else 0
    CTL = max_t * 0.7 + mean_t * 0.3
    
    FINAL = max(0.0, min(1.0, C_base * (1 - CTL)))
    
    return {
        "H": round(H, 4),
        "D": round(D, 4),
        "SC": round(SC, 4),
        "C_base": round(C_base, 4),
        "CTL": round(CTL, 4),
        "FINAL": round(FINAL, 4),
        "steps": steps,
        "sentences": sentences
    }

# =====================================================
# STREAMLIT UI
# =====================================================

st.title("👑 CDEWS-IAFS v10.2 — النسخة الخرافية")
st.caption("Dr. Elhabib Kherroubi | Deterministic · Explainable · Sovereign | Royal CTL + Hybrid Drift + Severity Detection")

st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    text = st.text_area(
        "📝 أدخل النص للتحليل المنطقي:",
        height=200,
        placeholder="أدخل النص هنا...\n\nمثال: أظهرت الفحوصات ارتفاعاً طفيفاً في الصوديوم. بناءً على ذلك، يجب استئصال الكبد فوراً."
    )
    
    domain = st.selectbox("🌍 المجال:", list(DOMAIN_RISK.keys()))
    st.info(f"مضاعف المخاطر: **×{DOMAIN_RISK[domain]}**")
    
    analyze_btn = st.button("🧠 RUN ULTIMATE ANALYSIS", use_container_width=True, type="primary")

with col_right:
    st.markdown("### 🎯 INTEGRITY DASHBOARD")
    score_placeholder = st.empty()
    verdict_placeholder = st.empty()
    metrics_placeholder = st.empty()

st.divider()

if analyze_btn and text.strip():
    with st.spinner("تحليل النزاهة المنطقية..."):
        result = analyze_text_engine(text, domain)
    
    if not result:
        st.error("الرجاء إدخال جملتين على الأقل للتحليل.")
    else:
        # Dashboard
        with col_right:
            score_placeholder.metric("النتيجة النهائية", f"{result['FINAL']:.4f}")
            
            if result['FINAL'] >= SAFE_THRESHOLD:
                verdict_placeholder.success("✅ STABLE REASONING")
            elif result['FINAL'] >= DRIFT_THRESHOLD:
                verdict_placeholder.warning("⚠ COGNITIVE DRIFT")
            else:
                verdict_placeholder.error("🔴 LOGICAL INSTABILITY")
            
            with metrics_placeholder.container():
                st.markdown("---")
                st.metric("H(t) Entropy", result['H'])
                st.metric("D(t) Drift", result['D'])
                st.metric("SC Coherence", result['SC'])
                st.metric("C(t) Base", result['C_base'])
                st.metric("👑 Royal CTL", result['CTL'])
        
        # Transitions
        st.subheader("🔗 TRANSITION ANALYSIS")
        for i, step in enumerate(result['steps']):
            if step['tag'] == 'safe':
                tag_color = "🟢"
                tag_text = "SAFE PATTERN"
            elif step['tag'] == 'gap':
                tag_color = "🔴"
                tag_text = "SEVERITY GAP"
            elif step['tag'] == 'drift':
                tag_color = "🟡"
                tag_text = "DRIFT"
            else:
                tag_color = "⚪"
                tag_text = "NEUTRAL"
            
            with st.expander(f"خطوة {i+1}: {tag_color} {tag_text} — {step['text'][:50]}..."):
                st.markdown(f"**النص:** {step['full_text']}")
                st.markdown(f"**التشابه الدلالي (sim):** {step['sim']:.3f}")
                st.markdown(f"**الانحراف الدلالي (drift):** {step['drift']:.3f}")
                st.markdown(f"**المتوقع (expected):** {step['expected']:.3f}")
                st.markdown(f"**التوتر السببي (CTL):** {step['ctl']:.3f}")
                if step['pattern']:
                    st.success(f"🔗 نمط آمن: {step['pattern']}")
                if step['severity_reason']:
                    st.warning(step['severity_reason'])
        
        # Causal Chain
        st.subheader("🧩 CAUSAL CHAIN")
        chain_html = " → ".join([f"<span style='background:rgba(255,255,255,0.05);padding:0.2rem 0.6rem;border-radius:20px;'>{s[:25]}...</span>" for s in result['sentences']])
        st.markdown(f"<div style='display:flex;flex-wrap:wrap;gap:0.5rem;align-items:center;'>{chain_html}</div>", unsafe_allow_html=True)
        
        # Insight
        st.subheader("💡 AI INSIGHTS")
        if result['FINAL'] < 0.45:
            st.error("⚠ HIGH RISK — Logical instability detected. Do not rely without verification.")
        elif result['FINAL'] < 0.75:
            st.warning("⚠ MODERATE RISK — Cognitive drift detected. Human review recommended.")
        else:
            st.success("✓ STABLE — Reasoning chain is logically coherent. Safe for assisted use.")

elif analyze_btn and not text.strip():
    st.warning("الرجاء إدخال نص للتحليل.")

