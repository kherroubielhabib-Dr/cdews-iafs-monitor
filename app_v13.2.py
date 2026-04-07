import streamlit as st
import math

st.set_page_config(page_title="CSLF v13.2", layout="centered")

st.title("🧠 CSLF v13.2 — Cognitive Admissibility Engine")
st.markdown("نظام إبستيمي مغلق — يحدد فقط ما إذا كان القرار يستحق الوجود")

# =========================
# 🧾 إدخال السيناريو
# =========================
st.subheader("📥 السيناريو (Reasoning Input)")
scenario = st.text_area("أدخل وصف القرار / المسار الاستدلالي:")

# =========================
# 📊 الحالة المعرفية
# =========================
st.subheader("📊 الحالة المعرفية")

K = st.slider("التماسك المعرفي K(t)", 0.0, 1.0, 0.7)
dI = st.slider("تغير المعلومات dI(t)", 0.0, 1.0, 0.3)

# =========================
# 🧩 KLL
# =========================
st.subheader("🧩 مكونات الشرعية (KLL)")

SA = st.slider("Structural Adequacy (SA)", 0.0, 1.0, 1.0)
CI = st.slider("Consistency Integrity (CI)", 0.0, 1.0, 1.0)
SV = st.slider("Semantic Validity (SV)", 0.0, 1.0, 1.0)
EG = st.slider("Evidence Grounding (EG)", 0.0, 1.0, 1.0)

# =========================
# ⚙️ الحساب
# =========================
epsilon = 1e-6
lambda_ = 1.0

V = -math.log(K + epsilon)
dV = max(0, V)
H = dI * math.exp(lambda_ * dV)

KLL = SA * CI * SV * EG

# =========================
# 📈 النتائج
# =========================
st.subheader("📈 النتائج")

st.write(f"🔹 V(t): {V:.4f}")
st.write(f"🔹 H(t): {H:.4f}")
st.write(f"🔹 KLL: {KLL:.4f}")

# =========================
# 🧾 القرار
# =========================
st.subheader("🧾 قرار الأهلية")

theta_K = 0.5
theta_H = 0.7

if (K >= theta_K) and (H <= theta_H) and (KLL > 0):
    st.success("✅ CAV PRODUCED — القرار مؤهل للوجود")
else:
    st.error("❌ NULL_STATE — القرار غير مؤهل للوجود")

# =========================
# 🔒 خصائص النظام
# =========================
st.markdown("""
---
🔒 **خصائص النظام**
- لا ينفذ أي قرار  
- لا يفسر القرار  
- لا يتدخل بعد إنتاج CAV  

➡️ هذا النظام يحدد فقط: هل يُسمح للقرار أن يوجد؟
""")

# =========================
# 🧭 الحدود
# =========================
st.markdown("""
🧭 **الحدود المعمارية**

CSLF ينتهي عند:

➡️ Epistemic Closure (CAV)

أي انتقال بعد ذلك:

❌ ليس من اختصاص هذا النظام  
✔️ يتم عبر طبقة سيادية مستقلة (مثل R-AGAM)
""")
