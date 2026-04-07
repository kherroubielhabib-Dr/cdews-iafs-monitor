import streamlit as st
import math

# ----------------------------
# إعداد الصفحة
# ----------------------------
st.set_page_config(page_title="CSLF v13.2", layout="centered")

st.title("🧠 CSLF v13.2 — Cognitive Admissibility Engine")
st.markdown("نظام إبستيمي مغلق — يحدد فقط ما إذا كان القرار يستحق الوجود")

# ----------------------------
# المدخلات
# ----------------------------
st.header("📊 الحالة المعرفية")

K = st.slider("التماسك المعرفي K(t)", 0.0, 1.0, 0.7)
dI = st.slider("تغير المعلومات dI(t)", 0.0, 1.0, 0.2)

st.header("🧩 مكونات الشرعية (KLL)")

SA = st.slider("Structural Adequacy (SA)", 0.0, 1.0, 1.0)
CI = st.slider("Consistency Integrity (CI)", 0.0, 1.0, 1.0)
SV = st.slider("Semantic Validity (SV)", 0.0, 1.0, 1.0)
EG = st.slider("Evidence Grounding (EG)", 0.0, 1.0, 1.0)

# ----------------------------
# الثوابت
# ----------------------------
epsilon = 1e-6
lambda_ = 2.0
theta_K = 0.6
theta_H = 0.5

# ----------------------------
# الحسابات
# ----------------------------
V = -math.log(K + epsilon)
dV = max(0, V - 0)  # baseline = 0

H = dI * math.exp(lambda_ * dV)

KLL = SA * CI * SV * EG

# ----------------------------
# النتائج
# ----------------------------
st.header("📈 النتائج")

st.write(f"🔹 V(t): {V:.4f}")
st.write(f"🔹 H(t): {H:.4f}")
st.write(f"🔹 KLL: {KLL:.4f}")

# ----------------------------
# منطق القرار (CSLF)
# ----------------------------
st.header("🧾 قرار الأهلية")

if KLL == 0:
    st.error("❌ FAILED_LEGITIMACY — انهيار الشرعية المعرفية")
elif H > theta_H:
    st.error("❌ FAILED_HAZARD — خطر ديناميكي مرتفع")
elif K < theta_K:
    st.warning("⚠️ BELOW_COHERENCE — التماسك غير كافٍ")
else:
    st.success("✅ CAV PRODUCED — القرار مؤهل للوجود (Epistemically Admissible)")

# ----------------------------
# ملاحظات معمارية
# ----------------------------
st.markdown("---")
st.markdown("""
### 🔒 خصائص النظام

- لا ينفذ أي قرار  
- لا يفسر القرار  
- لا يتدخل بعد إنتاج CAV  

➡️ هذا النظام يحدد فقط:
**هل يُسمح للقرار أن يوجد؟**

---

### 🧭 الحدود المعمارية

CSLF ينتهي عند:

➡️ Epistemic Closure (CAV)

أي انتقال بعد ذلك:

❌ ليس من اختصاص هذا النظام  
✔️ يتم عبر طبقة سيادية مستقلة (مثل R-AGAM)

---
""")
