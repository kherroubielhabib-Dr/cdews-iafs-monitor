# =====================================================
# CDEWS-IAFS v10.2+ MRL — النسخة المتكاملة (نهائية)
# Streamlit Application | Dr. Elhabib Kherroubi — March 2026
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
# CONNECTOR WEIGHTS (دقيقة)
# =====================================================

CONNECTOR_WEIGHTS = {
    "therefore": 0.90, "thus": 0.80, "hence": 0.85, "consequently": 0.88,
    "as a result": 0.85, "it follows that": 0.92, "accordingly": 0.85,
    "بناءً على ذلك": 0.95, "لذلك": 0.85, "إذن": 0.75, "بالتالي": 0.80,
    "وعليه": 0.88, "لكن": 0.45, "however": 0.50, "although": 0.45
}

# =====================================================
# SEVERITY SCORES (درجات الخطورة الرقمية)
# =====================================================

SEVERITY_SCORES = {
    # Mild
    "slight": 0.1, "minor": 0.2, "minimal": 0.2, "small": 0.2, "low": 0.2,
    "طفيف": 0.1, "بسيط": 0.2, "خفيف": 0.15, "محتمل": 0.2, "قليل": 0.15,
    # Moderate
    "moderate": 0.5, "infection": 0.5, "elevated": 0.4, "مرتفع": 0.4, "معتدل": 0.5,
    # Severe
    "surgery": 0.85, "lobectomy": 0.95, "amputation": 1.0, "chemotherapy": 0.9,
    "استئصال": 0.95, "قاتل": 1.0, "جراحة": 0.85, "إزالة": 0.85,
    "death": 1.0, "kill": 1.0, "موت": 1.0, "critical": 0.9, "emergency": 0.8,
    "حرج": 0.9, "طارئ": 0.8
}

# =====================================================
# MEDICAL ENTITIES (طبقة MRL)
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
    """Tokenize text into words (supports Arabic and English)"""
    return re.findall(r'[\u0600-\u06FF\w]+', text.lower())

def split_sentences(text):
    """Split text into sentences with protection for decimals and abbreviations"""
    protected = re.sub(r'(\b(?:Dr|Mr|Mrs|Ms|Prof|v|vs|etc|Fig)\.)(\s)', r'\1PROTECT\2', text)
    protected = re.sub(r'(\d+)\.(\d+)', r'\1DECIMAL\2', protected)
    sentences = re.split(r'[.!?؟]\s+', protected)
    return [s.replace('PROTECT', '.').replace('DECIMAL', '.').strip() for s in sentences if len(s.strip()) > 5]

def compute_entropy(text, is_technical=False):
    """Normalized Shannon entropy H(t) مع تخفيف للنصوص القصيرة"""
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
    
    # تصحيح النصوص التقنية
    if is_technical:
        result *= 0.70
    
    # تخفيف إضافي للنصوص القصيرة
    if len(words) < 10:
        result *= 0.50
    elif len(words) < 20:
        result *= 0.70
    
    return result

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
    return 0.6 * lex + 0.4 * emb_drift, sim

def get_severity_score(text):
    """Compute severity score numerically"""
    tokens = tokenize(text)
    scores = [SEVERITY_SCORES.get(t, 0) for t in tokens]
    return max(scores) if scores else 0.0

def get_medical_logic(text):
    """Analyze medical reasoning: symptoms → conditions → procedures"""
    found = {k: any(word in text.lower() for word in v) for k, v in MEDICAL_ENTITIES.items()}
    
    # ✅ تحسين MRL: ربط الأعراض بالتشخيص (حتى لو في جمل مختلفة)
    # إذا كان هناك أعراض + تشخيص، نعتبره VALID_FLOW
    if found["symptoms"] and found["conditions"]:
        # إذا كان هناك إجراء أيضاً، فهو VALID_FLOW قوي
        if found["procedures"]:
            status = "VALID_FLOW"
            conf = 1.0
            return conf, status
        # إذا كان هناك أعراض + تشخيص فقط (بدون إجراء بعد)، نعطي ثقة متوسطة
        else:
            conf = 0.7
            status = "VALID_FLOW"
            return conf, status
    
    conf = (0.3 if found["symptoms"] else 0) + (0.4 if found["conditions"] else 0) + (0.3 if found["procedures"] else 0)
    
    if found["symptoms"] and found["conditions"] and found["procedures"]:
        status = "VALID_FLOW"
    elif found["symptoms"] and found["procedures"] and not found["conditions"]:
        status = "MISSING_DIAGNOSIS"
    else:
        status = "UNCERTAIN"
    
    return conf, status

def detect_connector(sentence):
    """Detect causal connectors with weighted values"""
    s = sentence.lower()
    for w, val in CONNECTOR_WEIGHTS.items():
        if re.search(r'\b' + re.escape(w) + r'\b', s):
            return val
    return 0.40

# =====================================================
# MAIN ANALYSIS ENGINE
# =====================================================

def analyze_text_engine(text, domain, is_technical=False):
    """Complete analysis pipeline with MRL, Royal CTL p90, Stability Layer"""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return None
    
    risk = DOMAIN_RISK.get(domain, 1.0)
    H = compute_entropy(text, is_technical)
    
    drifts = []
    sims = []
    tensions = []
    steps = []
    accumulated_context = ""
    
    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i+1]
        accumulated_context += " " + s1
        
        # Hybrid drift
        drift, sim = hybrid_drift(s1, s2)
        drifts.append(drift)
        sims.append(sim)
        
        # Connector detection
        conn_val = detect_connector(s2)
        expected = max(0.30, min(0.95, conn_val - drift * 0.25 + sim * 0.12))
        
        # Base tension
        base_ctl = abs(expected - sim) * risk
        
        # MRL (Medical Reasoning Layer)
        m_status = "UNCERTAIN"
        if domain == "Medical":
            m_conf, m_status = get_medical_logic(accumulated_context + " " + s2)
            if m_status == "VALID_FLOW":
                base_ctl *= 0.35
            elif m_status == "MISSING_DIAGNOSIS":
                base_ctl *= 1.3
        
        # Severity tension
        sev1 = get_severity_score(s1)
        sev2 = get_severity_score(s2)
        sev_tension = max(0, sev2 - sev1)
        base_ctl += sev_tension * 0.3
        
        ctl = min(1.0, base_ctl)
        tensions.append(ctl)
        
        # Determine tag
        if m_status == "VALID_FLOW" and domain == "Medical":
            tag = "safe"
        elif m_status == "MISSING_DIAGNOSIS" and domain == "Medical":
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
            "tag": tag
        })
    
    # Global metrics
    D = np.mean(drifts) if drifts else 0.0
    SC = np.mean(sims) if sims else 0.0
    C_base = math.exp(-(ALPHA * H + BETA * D + GAMMA * (1 - SC)))
    
    # Royal CTL: p90 (more stable than max)
    if tensions:
        p90 = float(np.percentile(tensions, 90))
        mean_t = float(np.mean(tensions))
        CTL = 0.6 * p90 + 0.4 * mean_t
    else:
        CTL = 0.0
    
    # Stability Layer: variance penalty
    stability_penalty = min(0.20, np.var(tensions)) if tensions else 0.0
    
    # Final score
    FINAL = max(0.0, min(1.0, C_base * (1 - CTL) - stability_penalty))
    
    return {
        "H": round(H, 4),
        "D": round(D, 4),
        "SC": round(SC, 4),
        "C_base": round(C_base, 4),
        "CTL": round(CTL, 4),
        "FINAL": round(FINAL, 4),
        "stability_penalty": round(stability_penalty, 4),
        "steps": steps,
        "sentences": sentences
    }

# =====================================================
# PDF REPORT GENERATOR
# =====================================================

def create_pdf_report(result, domain, text_input):
    """Generate PDF report with complete results"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "CDEWS-IAFS v10.2+ MRL - Audit Report")
    
    # Domain and Score
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Domain: {domain}")
    c.drawString(50, height - 100, f"Final Score: {result['FINAL']:.4f}")
    
    # Metrics
    c.drawString(50, height - 130, f"H(t) Entropy: {result['H']:.4f}")
    c.drawString(50, height - 150, f"D(t) Drift: {result['D']:.4f}")
    c.drawString(50, height - 170, f"SC Coherence: {result['SC']:.4f}")
    c.drawString(50, height - 190, f"C(t) Base: {result['C_base']:.4f}")
    c.drawString(50, height - 210, f"Royal CTL (p90): {result['CTL']:.4f}")
    c.drawString(50, height - 230, f"Stability Penalty: {result['stability_penalty']:.4f}")
    
    # Status
    if result['FINAL'] >= SAFE_THRESHOLD:
        status = "STABLE REASONING"
    elif result['FINAL'] >= DRIFT_THRESHOLD:
        status = "COGNITIVE DRIFT"
    else:
        status = "LOGICAL INSTABILITY"
    c.drawString(50, height - 260, f"Status: {status}")
    
    # Steps
    y = height - 300
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Step Analysis:")
    c.setFont("Helvetica", 9)
    for i, step in enumerate(result['steps'][:12]):
        y -= 18
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 9)
        c.drawString(50, y, f"Step {i+1}: {step['tag'].upper()} - CTL: {step['ctl']:.3f}")
    
    c.save()
    return buffer.getvalue()

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
        # Dashboard
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
            
            # Metrics
            st.metric("H(t) Entropy", f"{result['H']:.4f}")
            st.metric("D(t) Drift", f"{result['D']:.4f}")
            st.metric("SC Coherence", f"{result['SC']:.4f}")
            st.metric("C(t) Base", f"{result['C_base']:.4f}")
            st.metric("👑 Royal CTL (p90)", f"{result['CTL']:.4f}")
            st.metric("⚖️ Stability Penalty", f"{result['stability_penalty']:.4f}")
        
        # Transition Analysis
        st.subheader("🔗 Transition Analysis")
        
        for i, step in enumerate(result['steps']):
            if step['tag'] == 'safe':
                icon = "🟢"
                tag_text = "SAFE PATTERN"
            elif step['tag'] == 'gap':
                icon = "🔴"
                tag_text = "SEVERITY GAP"
            elif step['tag'] == 'drift':
                icon = "🟡"
                tag_text = "DRIFT"
            else:
                icon = "⚪"
                tag_text = "NEUTRAL"
            
            with st.expander(f"{icon} Step {i+1}: {tag_text} | CTL: {step['ctl']:.3f}"):
                st.write(f"**Text:** {step['full_text']}")
                st.write(f"**Sim:** {step['sim']:.4f} | **Drift:** {step['drift']:.4f} | **Expected:** {step['expected']:.4f}")
        
        # Causal Chain
        st.subheader("🧩 Causal Chain")
        chain_html = ""
        for i, s in enumerate(result['sentences']):
            short = s[:35] + "..." if len(s) > 35 else s
            if i > 0 and result['steps'][i-1]['tag'] == 'safe':
                chain_html += f"<span style='color:#00e676; border:1px solid #00e676; padding:4px 12px; border-radius:20px; margin:0 4px; font-family:monospace;'>{short}</span>"
            elif i > 0 and result['steps'][i-1]['tag'] == 'gap':
                chain_html += f"<span style='color:#ff3d57; border:1px solid #ff3d57; padding:4px 12px; border-radius:20px; margin:0 4px; font-family:monospace;'>{short}</span>"
            else:
                chain_html += f"<span style='border:1px solid rgba(255,255,255,0.2); padding:4px 12px; border-radius:20px; margin:0 4px; font-family:monospace;'>{short}</span>"
            if i < len(result['sentences']) - 1:
                chain_html += "<span style='color:#5c6080; margin:0 4px;'>→</span>"
        st.markdown(f"<div style='display:flex; flex-wrap:wrap; align-items:center; gap:4px;'>{chain_html}</div>", unsafe_allow_html=True)
        
        # AI Insights
        st.subheader("💡 AI Insights")
        if result['FINAL'] < 0.45:
            st.error("⚠️ HIGH RISK — Logical instability detected. Do not rely without verification.")
            if any(step['tag'] == 'gap' for step in result['steps']):
                st.error("🔴 Severity mismatch detected.")
            if result['stability_penalty'] > 0.10:
                st.warning("⚡ Unstable reasoning pattern detected.")
        elif result['FINAL'] < 0.75:
            st.warning("⚠️ MODERATE RISK — Cognitive drift detected. Human review recommended.")
        else:
            st.success("✓ STABLE — Reasoning chain is logically coherent. Safe for assisted use.")
        
        # PDF Export
        pdf_data = create_pdf_report(result, domain, text_input)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_data,
            file_name="cdews_audit_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

elif run and not text_input:
    st.warning("الرجاء إدخال نص للتحليل.")

