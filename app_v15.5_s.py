# ============================================
# 🏛️ CSLF v15.5-S
# Sovereign Epistemic Gate (Production Ready)
# ============================================

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import hashlib
import json
import time
import streamlit as st


# ============================================
# 🔐 Epistemic Artifact (CAV)
# ============================================

@dataclass(frozen=True)
class CAV:
    decision_id: str
    epistemic_result: str
    context_hash: str
    signature: str
    timestamp: float
    validity_window: Optional[float]
    sealed: bool = True
    version: str = "CSLF-v15.5-S"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# ============================================
# 🧠 CSLF Engine
# ============================================

class CSLF:

    def evaluate(self, decision: Dict[str, Any], context: Dict[str, Any]) -> CAV:

        decision_id = self._generate_decision_id(decision, context)
        context_hash = self._hash_context(context)

        epistemic_valid = self._epistemic_filter(decision, context)
        result = "ALLOW_EXISTENCE" if epistemic_valid else "DENY_EXISTENCE"

        timestamp = time.time()
        signature = self._seal(decision_id, result, context_hash, timestamp)

        return CAV(
            decision_id=decision_id,
            epistemic_result=result,
            context_hash=context_hash,
            signature=signature,
            timestamp=timestamp,
            validity_window=None,
            sealed=True
        )

    # ============================================
    # 🧩 Epistemic Core
    # ============================================

    def _epistemic_filter(self, decision, context):

        if not self._check_structure(decision):
            return False

        if not self._check_coherence(decision):
            return False

        if not self._check_grounding(context):
            return False

        return True

    # ============================================
    # 🔍 Checks
    # ============================================

    def _check_structure(self, decision):
        required = {"premise", "inference", "conclusion"}
        return required.issubset(decision.keys()) and all(decision[k].strip() for k in required)

    def _check_coherence(self, decision):

        p = decision["premise"].lower()
        i = decision["inference"].lower()
        c = decision["conclusion"].lower()

        # ❌ Hard contradictions
        contradictions = [
            ("no infection", "antibiotic"),
            ("normal", "intervention"),
            ("no evidence", "treat"),
            ("no signs", "administer"),
        ]

        for neg, act in contradictions:
            if neg in p and act in c:
                return False

        # ❌ Empty logic
        if len(i.strip()) < 5 or len(c.strip()) < 5:
            return False

        # ❌ Identity collapse
        if p.strip() == c.strip():
            return False

        return True

    def _check_grounding(self, context):
        return isinstance(context, dict) and len(context) > 0

    # ============================================
    # 🔐 Sealing
    # ============================================

    def _seal(self, decision_id, result, context_hash, timestamp):
        payload = f"{decision_id}|{result}|{context_hash}|{timestamp}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _generate_decision_id(self, decision, context):
        raw = json.dumps(decision, sort_keys=True) + json.dumps(context, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _hash_context(self, context):
        return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


# ============================================
# 🚫 Safety Constraint
# ============================================

def assert_epistemic_purity(cav: CAV):
    assert cav.epistemic_result in ["ALLOW_EXISTENCE", "DENY_EXISTENCE"]
    assert cav.validity_window is None
    assert cav.sealed is True


# ============================================
# 🌐 Streamlit UI
# ============================================

def run_ui():
    st.set_page_config(page_title="CSLF v15.5-S", layout="centered")

    st.title("🏛️ CSLF v15.5-S — Sovereign Epistemic Gate")
    st.caption("Epistemic legitimacy only — No execution authority")

    st.markdown("### 📥 Decision Input")

    premise = st.text_area("Premise")
    inference = st.text_area("Inference")
    conclusion = st.text_area("Conclusion")

    st.markdown("### 📥 Context Input")
    context_str = st.text_area("Context JSON", value='{"patient_id": "123"}')

    if st.button("Evaluate Decision"):

        try:
            context = json.loads(context_str)

            decision = {
                "premise": premise,
                "inference": inference,
                "conclusion": conclusion
            }

            cslf = CSLF()
            cav = cslf.evaluate(decision, context)

            assert_epistemic_purity(cav)

            st.markdown("### 📦 CAV Output")
            st.code(cav.to_json(), language="json")

            if cav.epistemic_result == "ALLOW_EXISTENCE":
                st.success("Epistemically VALID")
            else:
                st.error("Epistemically INVALID")

        except Exception as e:
            st.error(f"Error: {str(e)}")


# ============================================
# 🚀 Entry Point
# ============================================

if __name__ == "__main__":
    run_ui()

