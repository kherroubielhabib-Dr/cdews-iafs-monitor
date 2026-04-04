import math, json, hashlib, hmac, secrets, uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SovereignConfig:
    LAMBDA: float = 5.5
    H_CRIT: float = 4.5
    KLL_MIN: float = 0.40
    THETA_K: float = 0.30
    WINDOW: int = 5
    DECAY: float = 0.85
    KLL_W: float = 0.25
    EPS: float = 1e-9

CFG = SovereignConfig()

# ════════════════════════════════════════════════════════════════
# SIGNING
# ════════════════════════════════════════════════════════════════

_SECRET = secrets.token_bytes(32)
PUBLIC_KEY_REF = "CSLF_" + hashlib.sha256(_SECRET).hexdigest()[:12]

def sign(payload: bytes) -> str:
    return hmac.new(_SECRET, payload, hashlib.sha256).hexdigest()

def verify(payload: bytes, sig: str) -> bool:
    return hmac.compare_digest(sign(payload), sig)

# ════════════════════════════════════════════════════════════════
# NLP CLASSIFIER
# ════════════════════════════════════════════════════════════════

LEXICON = {
    "chaos": ["random","invalid","noise","corrupt"],
    "evidence": ["data","evidence","proves","analysis","study"],
    "uncertainty": ["maybe","likely","possibly"]
}

def classify(text: str) -> str:
    t = text.lower()
    if any(w in t for w in LEXICON["chaos"]): return "chaos"
    if any(w in t for w in LEXICON["evidence"]): return "stable"
    if any(w in t for w in LEXICON["uncertainty"]): return "drift"
    return "drift"

def parse_steps(text: str) -> List[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]

# ════════════════════════════════════════════════════════════════
# CORE MATH
# ════════════════════════════════════════════════════════════════

DISTRIBUTIONS = {
    "stable": [0.78,0.14,0.05,0.03],
    "drift":  [0.50,0.25,0.15,0.10],
    "chaos":  [0.25,0.25,0.25,0.25],
}

def entropy(p):
    return -sum(x * math.log(x + CFG.EPS) for x in p)

def compute_sv(text: str) -> float:
    n = len(text.split())
    if n < 3:
        return 0.2
    return 0.9

# ════════════════════════════════════════════════════════════════
# ENGINE
# ════════════════════════════════════════════════════════════════

class CSLFv55:

    def __init__(self):
        self.K_hist = []
        self.V_prev = 0.0
        self.I_prev = 0.0
        self.trace = []
        self.sealed = False
        self.result = None

    def step(self, text: str):

        state = classify(text)
        p = DISTRIBUTIONS[state]

        Hs = entropy(p)
        K = max(CFG.EPS, 1 - Hs / math.log(len(p)))

        V = -math.log(K + CFG.EPS)
        dV = max(0, V - self.V_prev)
        self.V_prev = V

        I = min(1.0, K + (0.35 if state=="chaos" else 0.2 if state=="drift" else 0))
        dI = max(0, I - self.I_prev)
        self.I_prev = I

        raw_H = (dI + 0.05) * math.exp(CFG.LAMBDA * (dV + 0.02))
        H = min(CFG.H_CRIT*3, math.log1p(raw_H))

        self.K_hist.append(K)
        if len(self.K_hist) > CFG.WINDOW:
            self.K_hist.pop(0)

        weights = [CFG.DECAY**i for i in range(len(self.K_hist))][::-1]
        W = sum(weights)

        SA = sum(k*w for k,w in zip(self.K_hist,weights))/W
        CI = 1 - dV

        SV = compute_sv(text)

        EG = 1.0 if state=="stable" else 0.5 if state=="drift" else 0.2

        KLL = (SA * CI * SV * EG) ** CFG.KLL_W

        self.trace.append((K,H,KLL))

    def finalize(self):

        self.sealed = True

        weights = [CFG.DECAY**i for i in range(len(self.trace))][::-1]
        W = sum(weights)

        K_avg = sum(t[0]*w for t,w in zip(self.trace,weights))/W
        H_max = max(t[1] for t in self.trace)
        KLL = self.trace[-1][2]

        if H_max > CFG.H_CRIT:
            verdict = "FAILED_HAZARD"
        elif K_avg < CFG.THETA_K:
            verdict = "FAILED_COHERENCE"
        elif KLL < CFG.KLL_MIN:
            verdict = "FAILED_LEGITIMACY"
        else:
            verdict = "PASSED"

        self.result = {
            "verdict": verdict,
            "K": round(K_avg,4),
            "H": round(H_max,4),
            "KLL": round(KLL,4)
        }

        return self.result

    def seal(self):

        payload = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": self.result
        }

        canonical = json.dumps(payload, sort_keys=True).encode()
        return {
            **payload,
            "hash": hashlib.sha256(canonical).hexdigest(),
            "sig": sign(canonical),
            "pub": PUBLIC_KEY_REF
        }
