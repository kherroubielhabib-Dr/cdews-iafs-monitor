from dataclasses import dataclass
import math
import hashlib
import time
import json
from typing import Dict, Tuple, Any


# =========================
# 🧬 Epistemic Vector Model
# =========================

@dataclass(frozen=True)
class EpistemicVector:
    evidence: float   # E
    risk: float       # R
    intervention: float  # T

    def direction(self) -> float:
        """
        Semantic polarity / directional force
        """
        return self.intervention - self.evidence


# =========================
# 🧠 Encoder (Semantic → Vector)
# =========================

class SemanticEncoder:
    """
    Converts text into epistemic vector.
    NOTE: This is a deterministic heuristic placeholder.
    Replace later with embedding model (R-AGAM upgrade path).
    """

    evidence_keywords = [
        "no evidence", "normal", "benign", "stable",
        "no infection", "unremarkable", "negative"
    ]

    risk_keywords = [
        "infection", "fever", "elevated", "abnormal",
        "severe", "critical", "high"
    ]

    intervention_keywords = [
        "treat", "administer", "start", "give",
        "antibiotic", "therapy", "medication", "drug"
    ]

    @staticmethod
    def encode(text: str) -> EpistemicVector:
        t = text.lower()

        e = sum(1 for k in SemanticEncoder.evidence_keywords if k in t)
        r = sum(1 for k in SemanticEncoder.risk_keywords if k in t)
        i = sum(1 for k in SemanticEncoder.intervention_keywords if k in t)

        # normalization (avoid zero collapse)
        norm = max(e + r + i, 1)

        return EpistemicVector(
            evidence=e / norm,
            risk=r / norm,
            intervention=i / norm
        )


# =========================
# ⚖️ CSLF Core Engine
# =========================

class CSLF_v15_6_S:

    def __init__(self, tau: float = 0.25):
        self.tau = tau

    # -------------------------
    # Structural validation
    # -------------------------
    def _check_structure(self, decision: Dict[str, str]) -> bool:
        required = ["premise", "inference", "conclusion"]
        return all(
            k in decision and isinstance(decision[k], str) and decision[k].strip()
            for k in required
        )

    # -------------------------
    # Coherence boundary
    # -------------------------
    def _check_coherence(self, vp: EpistemicVector, vc: EpistemicVector) -> bool:
        dp = vp.direction()
        dc = vc.direction()
        return abs(dp - dc) <= self.tau

    # -------------------------
    # Sealing (integrity layer)
    # -------------------------
    def _seal(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    # -------------------------
    # Main epistemic evaluation
    # -------------------------
    def evaluate(self, decision: Dict[str, str], context: Dict[str, Any] = None):

        if not self._check_structure(decision):
            return self._build_output(decision, "DENY_EXISTENCE", "STRUCTURE_INVALID")

        premise_vec = SemanticEncoder.encode(decision["premise"])
        conclusion_vec = SemanticEncoder.encode(decision["conclusion"])

        allowed = self._check_coherence(premise_vec, conclusion_vec)

        result = "ALLOW_EXISTENCE" if allowed else "DENY_EXISTENCE"

        return self._build_output(decision, result, "OK")

    # -------------------------
    # Output builder
    # -------------------------
    def _build_output(self, decision, result, status):

        payload = {
            "premise": decision.get("premise", ""),
            "inference": decision.get("inference", ""),
            "conclusion": decision.get("conclusion", ""),
            "result": result,
            "status": status,
            "timestamp": time.time()
        }

        # إنشاء نسخة من payload لإضافة الملح (salt) للتوقيع
        payload_for_signing = payload.copy()
        payload_for_signing["salt"] = "CSLF-v15.6-S"

        return {
            "context_hash": self._seal(payload),
            "decision_id": hashlib.md5(str(payload).encode()).hexdigest(),
            "epistemic_result": result,
            "status": status,
            "sealed": True,
            "signature": self._seal(payload_for_signing),  # تم التصحيح هنا
            "timestamp": payload["timestamp"],
            "version": "CSLF-v15.6-S"
        }


# =========================
# 🧪 Example Execution
# =========================

if __name__ == "__main__":

    engine = CSLF_v15_6_S(tau=0.25)

    test_case = {
        "premise": "No evidence of infection and patient is stable",
        "inference": "Clinical picture is benign",
        "conclusion": "Start antibiotic therapy as precaution"
    }

    result = engine.evaluate(test_case)

    print(json.dumps(result, indent=2))
