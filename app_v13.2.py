import streamlit as st
import hashlib
import time
import math
import pandas as pd

# =============================================================
# CORE CSLF v13.2 ENGINE (المنطق الذي بنيناه)
# =============================================================

class CSLF_v13_2_Engine:
    def __init__(self):
        self.K_MIN = 0.7
        self.H_MAX = 0.4
        if 'edi_store' not in st.session_state:
            st.session_state.edi_store = {}

    def compute_metrics(self, trace):
        scores = [s['confidence'] for s in trace]
        kll = min(scores) if scores else 0
        hazard = 1 - (sum(scores) / len(scores)) if scores else 1
        return kll, hazard

    def check_integrity(self, trace):
        gaps = []
        valid_relations = ["CAUSES", "SUPPORTS", "JUSTIFIES", "DERIVED_FROM"]
        for i in range(len(trace) - 1):
            a, b = trace[i], trace[i+1]
            # Structural check
            if b.get("predecessor_id") != a.get("id"):
                gaps.append(f"Struc_Gap_{i+1}")
            # Semantic check
            if b.get("relation").upper() not in valid_relations:
                gaps.append(f"SIC_Violation_{i+1}")
        return gaps

    def get_friction(self, user_id):
        edi = st.session_state.edi_store.get(user_id, 0)
        return math.pow(3, int(edi))

# =============================================================
# STREAMLIT UI - الواجهة التفاعلية
# =============================================================

st.set_page_config(page_title="CSLF v13.2 Sovereign Core", layout="wide")

st.title("🏛️ CSLF v13.2: Epistemic Operating System")
st.markdown("---")

# Sidebar for User Context
with st.sidebar:
    st.header("👤 الهوية الإبستيمية")
    user_id = st.text_input("معرف المستخدم (User ID)", value="ARCHITECT_01")
    current_edi = st.session_state.edi_store.get(user_id, 0)
    st.metric("مؤشر الانحراف (EDI)", round(current_edi, 2))
    
    engine = CSLF_v13_2_Engine()
    friction = engine.get_friction(user_id)
    st.warning(f"معامل الاحتكاك الحالي: {int(friction)}x")

# Main Interface: Scenario Input
st.header("⚡ محاكاة المسار الاستدلالي")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("إدخال خطوات القرار")
    num_steps = st.number_input("عدد الخطوات", min_value=1, max_value=5, value=3)
    
    trace = []
    for i in range(num_steps):
        with st.expander(f"الخطوة {i+1}", expanded=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                content = st.text_input(f"المحتوى المعرفي {i+1}", key=f"c_{i}")
            with c2:
                conf = st.slider(f"الثقة (Confidence)", 0.0, 1.0, 0.5, key=f"s_{i}")
            with c3:
                rel = st.selectbox(f"العلاقة (Relation)", ["CAUSES", "SUPPORTS", "JUSTIFIES", "DERIVED_FROM", "NONE"], key=f"r_{i}")
            
            trace.append({
                "id": i+1,
                "content": content,
                "confidence": conf,
                "relation": rel,
                "predecessor_id": i if i > 0 else None
            })

with col2:
    st.subheader("التحكم السيادي")
    intent = st.selectbox("وسم النية (Intent Tag)", ["NORMAL", "EMERGENCY_PROTECTION", "SPECULATION", "STRATEGIC_DIVERGENCE"])
    override = st.checkbox("تفعيل التجاوز السيادي (Sovereign Override)")
    
    if st.button("تشغيل المحرك 🚀"):
        # 1. الميثاق الأصم (Meta-Law)
        if any(s['confidence'] > 1.0 for s in trace):
            st.error("VETO: انتهاك سقف الثقة (Meta-Law Violation)")
        else:
            # 2. الحسابات
            kll, hazard = engine.compute_metrics(trace)
            gaps = engine.check_integrity(trace)
            
            # 3. المنطق
            admissible = kll >= engine.K_MIN and hazard <= engine.H_MAX
            
            # 4. النتائج
            st.markdown("### 📊 نتائج التحليل")
            
            if admissible:
                st.success("✅ المسار مؤهل إبستيمياً (ADMISSIBLE)")
            elif override:
                st.warning("⚠️ تم التجاوز السيادي (OVERRIDDEN)")
                # تحديث الـ EDI كعقوبة للتجاوز أو الفجوات
                st.session_state.edi_store[user_id] = current_edi + 1 + (len(gaps) * 0.5)
            else:
                st.error("❌ المسار مرفوض (REJECTED)")
                st.session_state.edi_store[user_id] = current_edi + (len(gaps) * 0.5)

            # عرض المؤشرات
            m1, m2 = st.columns(2)
            m1.metric("KLL (الشرعية)", round(kll, 3))
            m2.metric("Hazard (الخطر)", round(hazard, 3))
            
            if gaps:
                st.write("🔍 **الفجوات المكتشفة:**", gaps)
            
            # 5. الختم الجنائي (FAL)
            st.markdown("---")
            st.subheader("🔒 الختم الجنائي (Forensic Seal)")
            payload = f"{user_id}|{kll}|{status if 'status' in locals() else 'PROCESSED'}|{time.time()}"
            seal = hashlib.sha256(payload.encode()).hexdigest()
            st.code(f"Signature: {seal}\nStatus: {'OVERRIDDEN' if override else 'NORMAL'}")

# Footer
st.markdown("---")
st.caption("CSLF v13.2 | Universal Epistemic Lexicon Core | 2026")
