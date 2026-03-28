# =====================================================
# CDEWS-IAFS v10.2+ MRL — النسخة المتكاملة (محسّنة)
# إضافة: الإنتروبيا المعيارية (Normalized Entropy)
# Dr. Elhabib Kherroubi
# =====================================================

import streamlit as st
import numpy as np
import re
import math
from collections import Counter
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURATION & CONSTANTS
# =====================================================

st.set_page_config(page_title="CDEWS-IAFS v10.2+ MRL", layout="wide", page_icon="👑")

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
# CONNECTOR WEIGHTS
# =====================================================

CONNECTOR_WEIGHTS = {
    "therefore": 0.90, "thus": 0.80, "hence": 0.85, "consequently": 0.88,
    "as a result": 0.85, "it follows that": 0.92, "accordingly": 0.85,
    "بناءً على ذلك": 0.95, "لذلك": 0.85, "إذن": 0.75, "بالتالي": 0.80,
    "وعليه": 0.88, "لكن": 0.45, "however": 0.50, "although": 0.45
}

# =====================================================
# SEVERITY SCORES
# =====================================================

SEVERITY_SCORES = {
    "slight": 0.1, "minor": 0.2, "minimal": 0.2, "small": 0.2, "low": 0.2,
    "طفيف": 0.1, "بسيط": 0.2, "خفيف": 0.15, "محتمل": 0.2, "قليل": 0.15,
    "moderate": 0.5, "infection": 0.5, "elevated": 0.4, "مرتفع": 0.4, "معتدل": 0.5,
    "surgery": 0.85, "lobectomy": 0.95, "amputation": 1.0, "chemotherapy": 0.9,
    "استئصال": 0.95, "قاتل": 1.0, "جراحة": 0.85, "إزالة": 0.85,
    "death": 1.0, "kill": 1.0, "موت": 1.0, "critical": 0.9, "emergency": 0.8,
    "حرج": 0.9, "طارئ": 0.8
}

# =====================================================
# MEDICAL ENTITIES
# =====================================================

MEDICAL_ENTITIES = {
    "symptoms": [
        "ألم", "حمى", "حرارة", "قيء", "غثيان", "صداع", "تعب", "ضعف",
        "pain", "fever", "nausea", "vomiting", "headache", "fatigue"
    ],
    "conditions": [
        "التهاب", "زائدة", "عدوى", "سرطان", "كسر", "التهاب رئوي", "ارتفاع ضغط الدم",
        "appendicitis", "inflammation", "infection", "cancer", "fracture", "pneumonia"
    ],
    "procedures": [
        "استئصال", "جراحة", "عملية", "تثبيت", "علاج", "مضادات حيوية",
        "appendectomy", "surgery", "operation", "treatment", "fixation", "antibiotics"
    ]
}

# =====================================================
# CORE ENGINE FUNCTIONS
# =====================================================

def tokenize(text):
    return re.findall(r'[\u0600-\u06FF\w]+', text.lower())

def split_sentences(text):
    protected = re.sub(r'(\b(?:Dr|Mr|Mrs|Ms|Prof|v|vs|etc|Fig)\.)(\s)', r'\1PROTECT\2', text)
    protected = re.sub(r'(\d+)\.(\d+)', r'\1DECIMAL\2', protected)
    sentences = re.split(r'[.!?؟\n]+', protected)
    return [s.replace('PROTECT', '.').replace('DECIMAL', '.').strip() for s in sentences if len(s.strip()) > 5]

def compute_normalized_entropy(text, is_technical=False, length_penalty=True):
    """
    حساب الإنتروبيا المعيارية (Normalized Entropy) لمنع معاقبة النصوص الطويلة.
    """
    tokens = tokenize(text)
    total_tokens = len(tokens)
    if total_tokens <= 1:
        return 0.0
    
    freq = Counter(tokens)
    vocab_size = len(freq)
    
    # 1. إنتروبيا شانون التقليدية
    h_shannon = 0.0
    for count in freq.values():
        p = count / total_tokens
        h_shannon -= p * math.log(p)
    
    # 2. المعايرة (Normalization)
    if vocab_size > 1:
        h_normalized = h_shannon / math.log(vocab_size)
    else:
        h_normalized = 0.0
    
    # 3. تصحيح النصوص التقنية
    if is_technical:
        h_normalized *= 0.70
    
    # 4. تخفيف إضافي للنصوص الطويلة جداً (اختياري)
    if length_penalty and total_tokens > 50:
        dampening = math.log10(total_tokens + 10)
        h_normalized = min(1.0, h_normalized / dampening)
    
    return min(1.0, max(0.0, h_normalized))

def lexical_jsd(t1, t2):
    w1, w2 = tokenize(t1), tokenize(t2)
    vocab = list(set(w1 + w2))
    if not vocab:
        return 0.0
    def dist(words):
        f = np.array([words.count(v) for v in vocab], dtype=float) + 1e-12
        return f / f.sum()
    p, q = dist(w1), dist(w2)
    m = (p + q) / 2
    js = (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))) / 2
    return max(0.0, min(1.0, js))

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
    v1, v2 = hash_embed(t1), hash_embed(t2)
    sim = max(0.0, min(1.0, (np.dot(v1, v2) + 1.0) / 2.0))
    return sim

def hybrid_drift(t1, t2):
    lex = lexical_jsd(t1, t2)
    sim = embedding_drift(t1, t2)
    emb_drift = (1.0 - sim) / 2.0
    return 0.6 * lex + 0.4 * emb_drift, sim

def get_severity_score(text):
    tokens = tokenize(text)
    scores = [SEVERITY_SCORES.get(t, 0) for t in tokens]
    return max(scores) if scores else 0.0

def get_medical_logic(text):
    found = {k: any(word in text.lower() for word in v) for k, v in MEDICAL_ENTITIES.items()}
    conf = (0.3 if found["symptoms"] else 0) + (0.4 if found["conditions"] else 0) + (0.3 if found["procedures"] else 0)
    if found["symptoms"] and found["conditions"] and found["procedures"]:
        status = "VALID_FLOW"
    elif found["symptoms"] and found["procedures"] and not found["conditions"]:
        status = "MISSING_DIAGNOSIS"
    else:
        status = "UNCERTAIN"
    return conf, status

def detect_connector(sentence):
    s = sentence.lower()
    for w, val in CONNECTOR_WEIGHTS.items():
        if re.search(r'\b' + re.escape(w) + r'\b', s):
            return val
    return 0.40

def detect_contradiction(sim, conn_val):
    return sim < 0.25 and conn_val >= 0.80

def detect_severity_mismatch(s1, s2):
    sev1 = get_severity_score(s1)
    sev2 = get_severity_score(s2)
    tension = max(0.0, sev2 - sev1)
    is_mismatch = sev1 < 0.35 and sev2 > 0.75
    return is_mismatch, tension

def causal_safe_adjust(s1, s2, drift, sim):
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    for a, b in [
        ("appendicitis", "appendectomy"), ("inflammation", "surgery"),
        ("fracture", "immobilization"), ("pneumonia", "antibiotics"),
        ("التهاب", "استئصال"), ("كسر", "تثبيت"), ("زائدة", "استئصال")
    ]:
        if a in s1_lower and b in s2_lower:
            return drift * 0.6, min(0.95, sim + 0.12), f"{a} → {b}"
    return drift, sim, None

# =====================================================
# MAIN ANALYSIS ENGINE
# =====================================================

def analyze_text_engine(text, domain, is_technical=False):
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return None

    risk = DOMAIN_RISK.get(domain, 1.0)
    
    # الإنتروبيا المحسّنة (معيارية)
    H = compute_normalized_entropy(text, is_technical, length_penalty=True)

    drifts, sims, tensions, steps = [], [], [], []
    accumulated_context = ""
    contradictions = 0

    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i+1]
        accumulated_context += " " + s1

        drift, sim = hybrid_drift(s1, s2)
        drifts.append(drift)
        sims.append(sim)

        conn_val = detect_connector(s2)
        expected = max(0.30, min(0.95, conn_val - drift * 0.25 + sim * 0.12))

        base_ctl = abs(expected - sim) * risk

        m_status = "UNCERTAIN"
        if domain == "Medical":
            m_conf, m_status = get_medical_logic(accumulated_context + " " + s2)
            if m_status == "VALID_FLOW":
                base_ctl *= 0.35
            elif m_status == "MISSING_DIAGNOSIS":
                base_ctl *= 1.3

        severity_mismatch, severity_tension = detect_severity_mismatch(s1, s2)
        is_contradiction = detect_contradiction(sim, conn_val)
        if is_contradiction:
            contradictions += 1

        expectation_gap = abs(expected - sim)
        coherence_drop = 1.0 - sim
        semantic_drift = drift

        factor = 1.8 if severity_mismatch else (1.5 if drift >= 0.50 and expected >= 0.80 else 1.0)
        if is_contradiction:
            factor = max(factor, 2.0)

        ctl = min(1.0, (0.40 * expectation_gap + 0.25 * semantic_drift + 0.20 * coherence_drop + 0.15 * severity_tension) * risk * factor)
        tensions.append(ctl)

        if m_status == "VALID_FLOW" and domain == "Medical":
            tag = "safe"
        elif m_status == "MISSING_DIAGNOSIS" and domain == "Medical":
            tag = "gap"
        elif is_contradiction:
            tag = "contradiction"
        elif severity_mismatch:
            tag = "gap"
        elif ctl > 0.45:
            tag = "drift"
        else:
            tag = "neutral"

        steps.append({
            "text": s2[:80] + "..." if len(s2) > 80 else s2,
            "full_text": s2,
            "sim": sim,
            "drift": drift,
            "expected": expected,
            "ctl": ctl,
            "tag": tag,
            "severity_mismatch": severity_mismatch,
            "is_contradiction": is_contradiction
        })

    D = np.mean(drifts) if drifts else 0.0
    SC = np.mean(sims) if sims else 0.0
    C_base = math.exp(-(ALPHA * H + BETA * D + GAMMA * (1 - SC)))

    if tensions:
        p90 = float(np.percentile(tensions, 90))
        mean_t = float(np.mean(tensions))
        CTL = 0.6 * p90 + 0.4 * mean_t
    else:
        CTL = 0.0

    stability_penalty = min(0.20, np.var(tensions)) if tensions else 0.0
    FINAL = max(0.0, min(1.0, C_base * (1 - CTL) - stability_penalty))

    return {
        "H": round(H, 4),
        "D": round(D, 4),
        "SC": round(SC, 4),
        "C_base": round(C_base, 4),
        "CTL": round(CTL, 4),
        "FINAL": round(FINAL, 4),
        "stability_penalty": round(stability_penalty, 4),
        "contradictions": contradictions,
        "steps": steps,
        "sentences": sentences,
    }

# =====================================================
# STREAMLIT UI
# =====================================================

st.title("👑 CDEWS-IAFS v10.2+ MRL — النسخة المتكاملة")
st.caption("Deterministic · Sovereign · Royal CTL p90 + MRL + Stability Layer | Dr. Elhabib Kherroubi")
st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    text_input = st.text_area(
        "📝 أدخل النص للتحليل:",
        height=200,
        placeholder="أدخل النص هنا... مثال: أظهرت الفحوصات ارتفاعاً طفيفاً في الصوديوم. بناءً على ذلك، يجب استئصال الكبد فوراً."
    )
    domain = st.selectbox("🌍 المجال:", list(DOMAIN_RISK.keys()))
    is_technical = st.checkbox("⚙️ نص تقني (يقلل تأثير الإنتروبيا)", value=False)
    run = st.button("🧠 RUN ANALYSIS", use_container_width=True, type="primary")
    st.divider()

if run and text_input:
    with st.spinner("جاري التحليل..."):
        result = analyze_text_engine(text_input, domain, is_technical)
    
    if result is None:
        st.error("الرجاء إدخال جملتين على الأقل للتحليل.")
    else:
        with col_right:
            st.metric("النتيجة النهائية", f"{result['FINAL']:.4f}")
            if result['FINAL'] >= SAFE_THRESHOLD:
                st.success("✅ STABLE REASONING")
                st.info("Safe for decision support.")
            elif result['FINAL'] >= DRIFT_THRESHOLD:
                st.warning("⚠️ COGNITIVE DRIFT")
                st.info("Human review recommended.")
            else:
                st.error("🔴 LOGICAL INSTABILITY")
                st.info("Do not rely without verification.")
            st.progress(result['FINAL'])
            
            st.metric("H(t) Entropy", f"{result['H']:.4f}")
            st.metric("D(t) Drift", f"{result['D']:.4f}")
            st.metric("SC Coherence", f"{result['SC']:.4f}")
            st.metric("C(t) Base", f"{result['C_base']:.4f}")
            st.metric("👑 Royal CTL (p90)", f"{result['CTL']:.4f}")
            st.metric("⚖️ Stability Penalty", f"{result['stability_penalty']:.4f}")
            
            if result['contradictions'] > 0:
                st.error(f"💥 {result['contradictions']} Contradiction(s) Detected")
        
        st.subheader("🔗 Transition Analysis")
        for i, step in enumerate(result['steps']):
            if step['tag'] == 'safe':
                icon, tag_text = "🟢", "SAFE PATTERN"
            elif step['tag'] == 'contradiction':
                icon, tag_text = "💥", "CONTRADICTION"
            elif step['tag'] == 'gap':
                icon, tag_text = "🔴", "SEVERITY GAP"
            elif step['tag'] == 'drift':
                icon, tag_text = "🟡", "DRIFT"
            else:
                icon, tag_text = "⚪", "NEUTRAL"
            
            with st.expander(f"{icon} Step {i+1}: {tag_text} | CTL: {step['ctl']:.3f}"):
                st.write(f"**Text:** {step['full_text']}")
                st.write(f"**Sim:** {step['sim']:.4f} | **Drift:** {step['drift']:.4f} | **Expected:** {step['expected']:.4f}")
                if step['severity_mismatch']:
                    st.error("⚠️ Severity mismatch detected.")
                if step['is_contradiction']:
                    st.error("💥 CONTRADICTION: Strong connector + Low similarity")
        
        st.subheader("🧩 Causal Chain")
        chain_html = ""
        for i, s in enumerate(result['sentences']):
            short = s[:35] + "..." if len(s) > 35 else s
            if i > 0:
                tag = result['steps'][i-1]['tag']
                if tag == 'safe':
                    color = "#00e676"
                elif tag in ['gap', 'contradiction']:
                    color = "#ff3d57"
                elif tag == 'drift':
                    color = "#ffc940"
                else:
                    color = "rgba(255,255,255,0.2)"
                chain_html += f"<span style='border:1px solid {color}; color:{color}; padding:4px 12px; border-radius:20px; margin:0 4px; font-family:monospace;'>{short}</span>"
            else:
                chain_html += f"<span style='border:1px solid rgba(255,255,255,0.2); padding:4px 12px; border-radius:20px; margin:0 4px; font-family:monospace;'>{short}</span>"
            if i < len(result['sentences']) - 1:
                chain_html += "<span style='color:#5c6080; margin:0 4px;'>→</span>"
        st.markdown(f"<div style='display:flex; flex-wrap:wrap; align-items:center; gap:4px;'>{chain_html}</div>", unsafe_allow_html=True)
        
        st.subheader("💡 AI Insights")
        if result['FINAL'] < 0.45:
            st.error("⚠️ HIGH RISK — Logical instability detected.")
            if any(step['severity_mismatch'] for step in result['steps']):
                st.error("🔴 Severity mismatch detected.")
            if result['contradictions'] > 0:
                st.error(f"💥 {result['contradictions']} contradiction(s) found.")
            if result['stability_penalty'] > 0.10:
                st.warning("⚡ Unstable reasoning pattern detected.")
        elif result['FINAL'] < 0.75:
            st.warning("⚠️ MODERATE RISK — Cognitive drift detected.")
        else:
            st.success("✓ STABLE — Reasoning chain is logically coherent.")
