import math, json, hashlib, hmac, secrets, uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict

# ════════════════════════════════════════════════════════════════
# CONFIG (IMMUTABLE)
# ════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CFG:
    LAMBDA: float = 5.5
    H_CRIT: float = 4.5
    KLL_MIN: float = 0.40
    THETA_K: float = 0.30
    WINDOW: int = 5
    DECAY: float = 0.85
    W: float = 0.25
    EPS: float = 1e-9

# ════════════════════════════════════════════════════════════════
# SIGNING (SOVEREIGN SEAL)
# ════════════════════════════════════════════════════════════════

_SECRET = secrets.token_bytes(32)
PUB = hashlib.sha256(_SECRET).hexdigest()[:12]

def _sign(b: bytes) -> str:
    return hmac.new(_SECRET, b, hashlib.sha256).hexdigest()

def _verify(b: bytes, s: str) -> bool:
    return hmac.compare_digest(_sign(b), s)

# ════════════════════════════════════════════════════════════════
# NLP (MINIMAL DETERMINISTIC)
# ════════════════════════════════════════════════════════════════

LEX = {
    "chaos": ["random","invalid","noise","corrupt"],
    "stable": ["data","evidence","analysis","proves","study"],
    "drift": ["maybe","likely","possibly"]
}

def classify(t: str) -> str:
    t = t.lower()
    if any(w in t for w in LEX["chaos"]): return "chaos"
    if any(w in t for w in LEX["stable"]): return "stable"
    if any(w in t for w in LEX["drift"]): return "drift"
    return "drift"

def split_steps(txt: str) -> List[str]:
    return [x.strip() for x in txt.splitlines() if x.strip()]

# ════════════════════════════════════════════════════════════════
# CORE DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════

DIST = {
    "stable": [0.78,0.14,0.05,0.03],
    "drift":  [0.50,0.25,0.15,0.10],
    "chaos":  [0.25,0.25,0.25,0.25],
}

def H_entropy(p):
    return -sum(x * math.log(x + CFG.EPS) for x in p)

def SV(text: str) -> float:
    return 0.9 if len(text.split()) >= 3 else 0.2

# ════════════════════════════════════════════════════════════════
# ENGINE (STATELESS INTERFACE / STATEFUL CORE)
# ════════════════════════════════════════════════════════════════

class CSLF:

    def __init__(self):
        self.K_hist = []
        self.V_prev = 0.0
        self.I_prev = 0.0
        self.trace = []
        self.states = []

    def step(self, text: str):

        s = classify(text)
        self.states.append(s)

        p = DIST[s]

        Hs = H_entropy(p)
        K = max(CFG.EPS, 1 - Hs / math.log(len(p)))

        V = -math.log(K + CFG.EPS)
        dV = max(0, V - self.V_prev)
        self.V_prev = V

        I = min(1.0, K + (0.35 if s=="chaos" else 0.2 if s=="drift" else 0))
        dI = max(0, I - self.I_prev)
        self.I_prev = I

        raw_H = (dI + 0.05) * math.exp(CFG.LAMBDA * (dV + 0.02))
        H = min(CFG.H_CRIT*3, math.log1p(raw_H))

        self.K_hist.append(K)
        if len(self.K_hist) > CFG.WINDOW:
            self.K_hist.pop(0)

        w = [CFG.DECAY**i for i in range(len(self.K_hist))][::-1]
        W = sum(w)

        SA = sum(k*wi for k,wi in zip(self.K_hist,w)) / W
        CI = max(0.0, 1 - dV)

        sv = SV(text)
        eg = 1.0 if s=="stable" else 0.5 if s=="drift" else 0.2

        KLL = (SA * CI * sv * eg) ** CFG.W

        self.trace.append((K,H,KLL))

    # ════════════════════════════════════════════════════════════
    # FINALIZE (FAC)
    # ════════════════════════════════════════════════════════════

    def finalize(self):

        w = [CFG.DECAY**i for i in range(len(self.trace))][::-1]
        W = sum(w)

        K_avg = sum(t[0]*wi for t,wi in zip(self.trace,w)) / W
        H_max = max(t[1] for t in self.trace)
        KLL = self.trace[-1][2]

        if H_max > CFG.H_CRIT:
            v = "FAILED_HAZARD"
        elif K_avg < CFG.THETA_K:
            v = "FAILED_COHERENCE"
        elif KLL < CFG.KLL_MIN:
            v = "FAILED_LEGITIMACY"
        else:
            v = "PASSED"

        S = K_avg * (1 - min(1.0, H_max/(CFG.H_CRIT*3))) * KLL

        return {
            "verdict": v,
            "K": round(K_avg,4),
            "H": round(H_max,4),
            "KLL": round(KLL,4),
            "S": round(S,4),
            "states": self.states
        }

    # ════════════════════════════════════════════════════════════
    # CAV (SEALED)
    # ════════════════════════════════════════════════════════════

    def seal(self, result: Dict):

        payload = {
            "id": "CAV-" + uuid.uuid4().hex[:8],
            "ts": datetime.now(timezone.utc).isoformat(),
            "res": result
        }

        canon = json.dumps(payload, sort_keys=True).encode()

        return {
            **payload,
            "hash": hashlib.sha256(canon).hexdigest(),
            "sig": _sign(canon),
            "pub": PUB
        }

# ════════════════════════════════════════════════════════════════
# ONE-LINE EXEC (GHOST ENTRYPOINT)
# ════════════════════════════════════════════════════════════════

def run(path: str):
    e = CSLF()
    for s in split_steps(path):
        e.step(s)
    r = e.finalize()
    return e.seal(r)
