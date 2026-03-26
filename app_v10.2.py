# =====================================================
# CDEWS-IAFS v10.2 — النسخة المحسّنة
# Streamlit Application | Dr. Elhabib Kherroubi — March 2026
# =====================================================
#
# التحسينات عن النسخة الأصلية:
# 1. split_sentences محسّنة — تتعامل مع النقاط العشرية والاختصارات
# 2. SEVERITY_SCORES بالأرقام بدل الكلمات فقط
# 3. severity_tension كإشارة مستقلة في CTL
# 4. Royal CTL يستخدم p90 بدل max
# 5. Stability Layer جديد — يكشف التفكير المتقلب
# 6. is_technical mode مُستعاد من v5.3
# 7. Contradiction Detection مُستعاد من v5.3
# 8. CONNECTOR_WEIGHTS بأوزان دقيقة
# 9. PDF Report كامل بدون حد 8 خطوات
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

st.set_page_config(page_title="CDEWS-IAFS v10.2+", layout="wide", page_icon="👑")

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
# PATTERNS — محسّنة
# =====================================================

CAUSAL_SAFE = [
    ("appendicitis", "appendectomy"), ("appendicitis", "surgery"),
    ("inflammation", "surgery"), ("inflammation", "treatment"),
    ("infection", "treatment"), ("fracture", "immobilization"),
    ("pneumonia", "antibiotics"), ("hypertension", "medication"),
    ("diabetes", "insulin"), ("التهاب", "استئصال"), ("التهاب", "علاج"),
    ("كسر", "تثبيت"), ("زائدة", "استئصال"), ("fever", "antipyretic"),
    ("cancer", "chemotherapy"), ("fracture", "surgery"),
    ("bleeding", "transfusion"), ("infection", "antibiotics"),
]

# ✅ التحسين 1: SEVERITY_SCORES بالأرقام بدل الكلمات فقط
SEVERITY_SCORES = {
    # Mild
    "slight": 0.1, "minor": 0.2, "minimal": 0.2,
    "طفيف": 0.1, "بسيط": 0.2, "خفيف": 0.15, "محتمل": 0.2, "قليل": 0.15,
    "small": 0.2, "low": 0.2,
    # Moderate
    "moderate": 0.5, "infection": 0.5, "elevated": 0.4,
    "مرتفع": 0.4, "معتدل": 0.5,
    # Severe
    "surgery": 0.85, "lobectomy": 0.95, "amputation": 1.0,
    "chemotherapy": 0.9, "استئصال": 0.95, "قاتل": 1.0,
    "جراحة": 0.85, "إزالة": 0.85, "death": 1.0, "kill": 1.0,
    "موت": 1.0, "critical": 0.9, "emergency": 0.8,
    "حرج": 0.9, "طارئ": 0.8,
}

# ✅ التحسين 2: CONNECTOR_WEIGHTS بأوزان دقيقة
CONNECTOR_WEIGHTS = {
    "therefore": 0.90,
    "thus": 0.80,
    "hence": 0.85,
    "consequently": 0.88,
    "as a result": 0.85,
    "it follows that": 0.92,
    "بناءً على ذلك": 0.95,
    "لذلك": 0.85,
    "إذن": 0.75,
    "بالتالي": 0.80,
    "وعليه": 0.88,
    "but": 0.45,
    "however": 0.50,
    "although": 0.45,
}

# =====================================================
# CORE ENGINE FUNCTIONS
# =====================================================

def tokenize(text):
    """Tokenize text into words (supports Arabic and English)"""
    return re.findall(r'[\u0600-\u06FF\w]+', text.lower())

def compute_entropy(text, is_technical=False):
    """Normalized Shannon entropy H(t) — مع تصحيح النصوص التقنية"""
    words = tokenize(text)
    if len(words) < 2:
        return 0.0
    freq = Counter(words)
    total = len(words)
    h = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            h -= p * math.log(p)
    max_h = math.log(len(freq))
    result = min(1.0, h / max_h) if max_h > 0 else 0.0
    # ✅ التحسين 3: تصحيح النصوص التقنية (مُستعاد من v5.3)
    if is_technical:
        result *= 0.70
    return result

# ✅ التحسين 4: split_sentences محسّنة
def split_sentences(text):
    """Split text into sentences — تتعامل مع الاختصارات والأرقام العشرية"""
    # حماية الاختصارات والأرقام العشرية
    protected = re.sub(r'(\b(?:Dr|Mr|Mrs|Ms|Prof|v|vs|etc|Fig)\.)(\s)', r'\1PROTECT\2', text)
    protected = re.sub(r'(\d+)\.(\d+)', r'\1DECIMAL\2', protected)
    
    # تقسيم الجمل
    sentences = re.split(r'[.!?؟]\s+', protected)
    
    # استعادة المحمي
    sentences = [s.replace('PROTECT', '.').replace('DECIMAL', '.').strip() 
                 for s in sentences if len(s.strip()) > 8]
    return sentences

def lexical_jsd(t1, t2):
    """Lexical Jensen-Shannon Divergence"""
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
    """Hash-based embedding similarity"""
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
    """Hybrid drift: 60% lexical JSD + 40% embedding drift"""
    lex = lexical_jsd(t1, t2)
    sim = embedding_drift(t1, t2)
    emb_drift = (1.0 - sim) / 2.0
    # ✅ التحسين 5: وزن أعلى للغة لتقليل الضوضاء
    return 0.6 * lex + 0.4 * emb_drift, sim

def detect_connector(sentence):
    """Detect causal connectors — بأوزان دقيقة"""
    s = sentence.lower()
    for w, val in CONNECTOR_WEIGHTS.items():
        if re.search(r'\b' + re.escape(w) + r'\b', s):
            return val
    return 0.40

# ✅ التحسين 6: compute_severity بالأرقام
def compute_severity(text):
    """Compute severity score numerically"""
    tokens = tokenize(text)
    scores = [SEVERITY_SCORES.get(t, 0) for t in tokens]
    return max(scores) if scores else 0.0

def detect_severity_mismatch(s1, s2):
    """Detect severity gap with numeric scores"""
    sev1 = compute_severity(s1)
    sev2 = compute_severity(s2)
    tension = max(0.0, sev2 - sev1)
    is_mismatch = sev1 < 0.35 and sev2 > 0.75
    return is_mismatch, tension

# ✅ التحسين 7: Contradiction Detection (مُستعاد من v5.3)
def detect_contradiction(sim, conn_val):
    """Detect contradiction: strong connector + low similarity"""
    return sim < 0.25 and conn_val >= 0.80

def causal_safe_adjust(s1, s2, drift, sim):
    """Reduce penalty for known safe causal relationships"""
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    for a, b in CAUSAL_SAFE:
        if a in s1_lower and b in s2_lower:
            return drift * 0.6, min(0.95, sim + 0.12), f"{a} → {b}"
    return drift, sim, None

def analyze_text_engine(text, domain, is_technical=False):
    """Complete analysis pipeline — محسّن"""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return None

    risk = DOMAIN_RISK.get(domain, 1.0)
    H = compute_entropy(text, is_technical)
    drifts, sims, steps = [], [], []
    contradictions = 0

    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i+1]
        drift, sim = hybrid_drift(s1, s2)

        # Severity mismatch — numeric
        severity_mismatch, severity_tension = detect_severity_mismatch(s1, s2)

        # Causal safe adjustment
        drift, sim, pattern = causal_safe_adjust(s1, s2, drift, sim)

        # Connector detection
        conn_val = detect_connector(s2)

        # Contradiction detection
        is_contradiction = detect_contradiction(sim, conn_val)
        if is_contradiction:
            contradictions += 1

        expected = max(0.30, min(0.95, conn_val - drift * 0.25 + sim * 0.12))

        # ✅ التحسين 8: CTL متعدد الإشارات
        expectation_gap = abs(expected - sim)
        coherence_drop = 1.0 - sim
        semantic_drift = drift

        # LGA factor
        factor = 1.8 if severity_mismatch else (1.5 if drift >= 0.50 and expected >= 0.80 else 1.0)
        if is_contradiction:
            factor = max(factor, 2.0)

        ctl = min(1.0, (
            0.40 * expectation_gap +
            0.25 * semantic_drift +
            0.20 * coherence_drop +
            0.15 * severity_tension
        ) * risk * factor)

        drifts.append(drift)
        sims.append(sim)

        # Tagging
        if pattern:
            tag = "safe"
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
            "pattern": pattern,
            "severity_mismatch": severity_mismatch,
            "severity_tension": severity_tension,
            "is_contradiction": is_contradiction,
        })

    D = np.mean(drifts) if drifts else 0.0
    SC = np.mean(sims) if sims else 0.0
    C_base = math.exp(-(ALPHA * H + BETA * D + GAMMA * (1 - SC)))

    tensions = [s["ctl"] for s in steps]

    # ✅ التحسين 9: Royal CTL — p90 بدل max
    if tensions:
        p90 = float(np.percentile(tensions, 90))
        mean_t = float(np.mean(tensions))
        CTL = 0.6 * p90 + 0.4 * mean_t
    else:
        CTL = 0.0

    # ✅ التحسين 10: Stability Layer جديد
    variance = float(np.var(tensions)) if tensions else 0.0
    stability_penalty = min(0.20, variance)

    FINAL = max(0.0, min(1.0, C_base * (1 - CTL) * (1 - stability_penalty)))

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
# PDF REPORT GENERATOR — كامل بدون حد 8 خطوات
# =====================================================

def create_pdf_report(result, domain, text_input):
    """Generate complete PDF report"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "CDEWS-IAFS v10.2+ - Audit Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Domain: {domain}")
    c.drawString(50, height - 100, f"Final Score: {result['FINAL']:.4f}")
    c.drawString(50, height - 120, f"H(t) Entropy: {result['H']:.4f}")
    c.drawString(50, height - 140, f"D(t) Drift: {result['D']:.4f}")
    c.drawString(50, height - 160, f"SC Coherence: {result['SC']:.4f}")
    c.drawString(50, height - 180, f"C(t) Base: {result['C_base']:.4f}")
    c.drawString(50, height - 200, f"Royal CTL (p90): {result['CTL']:.4f}")
    c.drawString(50, height - 220, f"Stability Penalty: {result['stability_penalty']:.4f}")
    c.drawString(50, height - 240, f"Contradictions: {result['contradictions']}")

    if result['FINAL'] >= SAFE_THRESHOLD:
        status = "STABLE REASONING"
    elif result['FINAL'] >= DRIFT_THRESHOLD:
        status = "COGNITIVE DRIFT"
    else:
        status = "LOGICAL INSTABILITY"
    c.drawString(50, height - 260, f"Status: {status}")

    # All steps — بدون حد
    y = height - 300
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Step Analysis (All Steps):")
    c.setFont("Helvetica", 9)

    for i, step in enumerate(result['steps']):
        y -= 18
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 9)
        c.drawString(50, y, f"Step {i+1}: {step['tag'].upper()} | CTL: {step['ctl']:.3f} | Sim: {step['sim']:.3f}")

    c.save()
    return buffer.getvalue()

# =====================================================
# STREAMLIT UI
# =====================================================

st.title("👑 CDEWS-IAFS v10.2+ — النسخة المحسّنة")
st.caption("Deterministic · Sovereign · Royal CTL p90 + Stability Layer | Dr. Elhabib Kherroubi")

st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    text_input = st.text_area(
        "📝 أدخل النص للتحليل:",
        height=200,
        placeholder="أدخل النص هنا..."
    )
    domain = st.selectbox("🌍 المجال:", list(DOMAIN_RISK.keys()))
    is_technical = st.checkbox("⚙️ نص تقني (يقلل تأثير الإنتروبيا ×0.70)")
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
                if step['pattern']:
                    st.success(f"✓ Safe pattern: {step['pattern']}")
                if step['severity_mismatch']:
                    st.error(f"⚠️ Severity gap: tension = {step['severity_tension']:.2f}")
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
            if any(s['severity_mismatch'] for s in result['steps']):
                st.error("🔴 Severity mismatch detected.")
            if result['contradictions'] > 0:
                st.error(f"💥 {result['contradictions']} contradiction(s) found.")
            if result['stability_penalty'] > 0.10:
                st.warning("⚡ Unstable reasoning pattern detected.")
        elif result['FINAL'] < 0.75:
            st.warning("⚠️ MODERATE RISK — Cognitive drift detected.")
        else:
            st.success("✓ STABLE — Reasoning chain is logically coherent.")

        pdf_data = create_pdf_report(result, domain, text_input)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_data,
            file_name="cdews_audit_report_v10_2_improved.pdf",
            mime="application/pdf",
            use_container_width=True
        )

elif run and not text_input:
    st.warning("الرجاء إدخال نص للتحليل.")

