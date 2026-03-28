# =====================================================
# 👑 CDEWS-IAFS v11.0 MRL — Sovereign Edition (FINAL FIX)
# Cognitive Drift Early Warning System
# Deterministic · Sovereign · Non-Zero Collapse Logic
# Dr. Elhabib Kherroubi — March 2026
# =====================================================

import streamlit as st
import numpy as np
import re
import math
from collections import Counter
import warnings

# إعدادات الواجهة والتحذيرات
warnings.filterwarnings('ignore')
st.set_page_config(page_title="CDEWS-IAFS v11.0 MRL", layout="wide", page_icon="👑")

# =====================================================
# CONFIGURATION & HYPER-PARAMETERS
# =====================================================
SAFE_THRESHOLD = 0.75
DRIFT_THRESHOLD = 0.45

# أوزان الاستقرار الأساسية (Lyapunov Basis)
BASE_ALPHA = 0.45  # Entropy weight
BASE_BETA = 0.35   # Drift weight
BASE_GAMMA = 0.20  # Coherence weight

# معاملات الخطورة السيادية للمجالات
DOMAIN_RISK = {
    "General": 1.0,
    "Medical": 1.8,
    "Finance": 1.5,
    "Legal": 1.6,
    "AI Safety": 1.7,
}

# روابط الاستدلال وأوزانها المنطقية
CONNECTOR_WEIGHTS = {
    "therefore": 0.90, "thus": 0.80, "hence": 0.85, "consequently": 0.88,
    "as a result": 0.85, "it follows that": 0.92, "accordingly": 0.85,
    "بناءً على ذلك": 0.95, "لذلك": 0.85, "إذن": 0.75, "بالتالي": 0.80,
    "وعليه": 0.88, "لكن": 0.45, "however": 0.50, "although": 0.45
}

# قاموس شدة المصطلحات (Severity Map)
SEVERITY_SCORES = {
    "slight": 0.1, "minor": 0.2, "low": 0.2, "خفيف": 0.15, "بسيط": 0.2,
    "infection": 0.5, "moderate": 0.5, "مرتفع": 0.4, "elevated": 0.4,
    "جراحة": 0.85, "surgery": 0.85, "استئصال": 0.95, "قسطرة": 0.80,
    "critical": 0.9, "حرج": 0.9, "طارئ": 0.8, "emergency": 0.8,
    "death": 1.0, "موت": 1.0, "نزيف": 0.85, "hemorrhage": 0.9
}

# =====================================================
# CORE UTILITIES (TOKENIZATION & ENTROPY)
# =====================================================
def tokenize(text):
    """تحليل النص إلى وحدات منطقية تدعم العربية والإنجليزية."""
    return re.findall(r'[\u0600-\u06FF\w]+', text.lower())

def split_sentences(text):
    """تقسيم النص مع حماية الأرقام العشرية والاختصارات."""
    protected = re.sub(r'(\d+)\.(\d+)', r'\1DECIMAL\2', text)
    sentences = re.split(r'[.!?؟\n]+', protected)
    return [s.replace('DECIMAL', '.').strip() for s in sentences if len(s.strip()) > 5]

def compute_entropy(text):
    """حساب الإنتروبيا المعلوماتية (Information Entropy)."""
    tokens = tokenize(text)
    if len(tokens) < 2: return 0.0
    freq = Counter(tokens)
    h = 0.0
    for c in freq.values():
        p = c / len(tokens)
        h -= p * math.log(p)
    return min(1.0, h / math.log(len(freq)))

# =====================================================
# DRIFT & COHERENCE (JSD + HASH EMBEDDING)
# =====================================================
def lexical_jsd(t1, t2):
    """حساب تباعد جينسن-شانون المعجمي."""
    w1, w2 = tokenize(t1), tokenize(t2)
    vocab = list(set(w1 + w2))
    if not vocab: return 0.0
    def dist(words):
        f = np.array([words.count(v) for v in vocab], float) + 1e-12
        return f / f.sum()
    p, q = dist(w1), dist(w2)
    m = (p + q) / 2
    return float((np.sum(p*np.log(p/m)) + np.sum(q*np.log(q/m))) / 2)

def embedding_sim(t1, t2, dim=64):
    """محاكاة التشابه الدلالي عبر التشفير الرقمي (Hash Embedding Sim)."""
    vec = lambda txt: np.array([hash(w)%dim for w in tokenize(txt)])
    v1, v2 = vec(t1), vec(t2)
    if len(v1) == 0 or len(v2) == 0: return 0.0
    return np.mean(np.isin(v1, v2))

def hybrid_drift(t1, t2):
    """دمج الانزياح المعجمي والدلالي."""
    jsd = lexical_jsd(t1, t2)
    sim = embedding_sim(t1, t2)
    return 0.6 * jsd + 0.4 * (1 - sim), sim

# =====================================================
# SEVERITY ANALYSERS
# =====================================================
def get_severity(text):
    tokens = tokenize(text)
    return max([SEVERITY_SCORES.get(t, 0) for t in tokens], default=0)

def severity_gap(s1, s2):
    """كشف الفجوات المنطقية في شدة الإجراءات (مثلاً: صداع -> جراحة)."""
    return (get_severity(s2) - get_severity(s1)) > 0.5

# =====================================================
# THE ENGINE: CTL & STABILITY
# =====================================================
def compute_ctl(sentences, domain):
    """حساب التوتر السببي الملكي (Royal Causal Tension)."""
    risk_multiplier = DOMAIN_RISK.get(domain, 1.0)
    tensions, drifts = [], []

    for i in range(len(sentences)-1):
        s1, s2 = sentences[i], sentences[i+1]
        drift, sim = hybrid_drift(s1, s2)
        drifts.append(drift)

        # رصد الروابط
        conn = next((v for k,v in CONNECTOR_WEIGHTS.items() if k in s2.lower()), 0.4)
        expected = max(0.3, min(0.95, conn - drift*0.2 + sim*0.1))
        
        ctl = abs(expected - sim) * risk_multiplier
        if severity_gap(s1, s2): ctl *= 1.8 # عقوبة القفز غير المبرر في الخطورة

        tensions.append(min(1.0, ctl))

    if not tensions: return 0, [], []
    # الاستحواذ على p90 لضمان رصد أسوأ انقطاع منطقي
    p90 = np.percentile(tensions, 90)
    return 0.6 * p90 + 0.4 * np.mean(tensions), tensions, drifts

def calculate_stability(H, D, SC, ctl, domain_factor):
    """
    الدالة النيبرية للاستقرار السيادي.
    تستخدم Exponential Decay لضمان عدم الوصول للصفر المطلق.
    """
    instability = (BASE_ALPHA * H) + (BASE_BETA * D) + (BASE_GAMMA * (1 - SC))
    domain_scaled = 1 + (domain_factor - 1) * 0.3
    risk = (ctl**1.5) * domain_scaled
    
    # دالة الاستقرار: C(t) = e^(-instability - risk)
    return math.exp(-(instability + risk))

def compute_penalty(tensions, ctl):
    """حساب عقوبة التذبذب (Variance Penalty)."""
    if len(tensions) < 2: return 0.0
    var = np.var(tensions)
    return min(0.15, var * (0.5 + 0.5 * ctl))

def analyze(text, domain):
    """محرك التحليل الرئيسي."""
    sentences = split_sentences(text)
    if len(sentences) < 2: return None

    H = compute_entropy(text)
    ctl, tensions, drifts = compute_ctl(sentences, domain)
    
    D = np.mean(drifts) if drifts else 0
    SC = 1 - D # التماسك الهيكلي

    C_base = calculate_stability(H, D, SC, ctl, DOMAIN_RISK[domain])
    penalty = compute_penalty(tensions, ctl)

    # التطبيق النهائي للعقوبة كمعامل خفض (Attenuation) وليس طرحاً
    FINAL = C_base * (1 - penalty)

    return {
        "FINAL": FINAL, "H": H, "D": D, "SC": SC, 
        "CTL": ctl, "penalty": penalty, "sentences": sentences
    }

# =====================================================
# 👑 STREAMLIT SOVEREIGN INTERFACE
# =====================================================
st.markdown("<h1 style='text-align: center; color: #D4AF37;'>👑 CDEWS-IAFS v11.0 MRL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Sovereign Edition | Dr. Elhabib Kherroubi</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    input_text = st.text_area("📝 أدخل النص السيادي للتحليل:", height=250)
    selected_domain = st.selectbox("🌍 مجال السياسة (Policy Domain):", list(DOMAIN_RISK.keys()))
    run_btn = st.button("🧠 إجراء التحليل الحتمي", use_container_width=True)

if run_btn and input_text:
    res = analyze(input_text, selected_domain)
    
    if res:
        with col2:
            st.metric("النتيجة النهائية (Stability Index)", f"{res['FINAL']:.4f}")
            
            if res['FINAL'] >= SAFE_THRESHOLD:
                st.success("✅ بنية مستقرة")
            elif res['FINAL'] >= DRIFT_THRESHOLD:
                st.warning("⚠️ انزياح إدراكي")
            else:
                st.error("🔴 انهيار هيكلي")

            st.progress(res['FINAL'])
            
            with st.expander("📊 التفاصيل الرياضية"):
                st.write(f"**CTL (التوتر):** {res['CTL']:.4f}")
                st.write(f"**Entropy (الإنتروبيا):** {res['H']:.4f}")
                st.write(f"**Drift (الانزياح):** {res['D']:.4f}")
                st.write(f"**Penalty (العقوبة):** {res['penalty']:.4f}")

        st.subheader("🔗 تسلسل الاستدلال (Causal Chain)")
        cols = st.columns(len(res['sentences']))
        for i, s in enumerate(res['sentences']):
            cols[i].info(f"جملة {i+1}\n\n{s}")

    else:
        st.error("❌ النص قصير جداً. يرجى إدخال جملتين على الأقل.")

st.divider()
st.caption("CDEWS-IAFS v11.0 MRL — Sovereign Edition | 2026")
