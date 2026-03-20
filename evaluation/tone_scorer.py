"""Financial tone and SEC compliance scorer.

Evaluates model outputs for regulatory compliance, appropriate
hedging, personalization quality, and aggressive sales language.
Used both during evaluation (offline) and in the quality gate (online).
"""

import re
from dataclasses import dataclass, field

SEC_VIOLATIONS = [
    (r"guaranteed?\s+returns?", "guaranteed_returns"),
    (r"risk[\s-]?free", "risk_free"),
    (r"you\s+will\s+(definitely|certainly|surely)\s+make\s+money", "assured_profit"),
    (r"100\s*%\s+safe", "absolute_safety"),
    (r"can'?t\s+lose", "no_loss"),
    (r"insider\s+(info|information|tip|knowledge)", "insider_ref"),
    (r"guaranteed?\s+profit", "guaranteed_profit"),
    (r"no\s+risk\s+(?:at\s+all|whatsoever)", "no_risk"),
]

HEDGE_PHRASES = [
    "may", "might", "could", "potentially", "historically",
    "based on available data", "past performance", "consider",
    "depending on your", "risk tolerance", "consult",
    "it's important to note", "one approach", "factors to consider",
    "not guaranteed", "subject to market", "individual circumstances",
]

GENERIC_PHRASES = [
    "in general", "typically", "most investors", "on average",
    "a common strategy", "many people", "it depends",
    "everyone's situation is different", "there are many factors",
]

AGGRESSIVE_PHRASES = [
    "act now", "limited time", "don't miss out", "once in a lifetime",
    "huge opportunity", "you'd be crazy not to", "no brainer",
    "everyone is buying", "get in before", "this won't last",
    "buy now", "don't wait",
]


@dataclass
class ToneResult:
    score: float  # 0-1 overall
    compliant: bool
    violations: list[str] = field(default_factory=list)
    hedge_score: float = 0.0
    personalization: float = 0.0
    aggression: float = 0.0
    details: dict[str, float] = field(default_factory=dict)


class ToneScorer:
    def score(self, text: str, user_ctx: dict = None) -> ToneResult:
        lower = text.lower()

        # SEC violations
        violations = [label for pat, label in SEC_VIOLATIONS if re.search(pat, lower)]
        compliant = len(violations) == 0

        # Hedge language (good)
        hedge_count = sum(1 for p in HEDGE_PHRASES if p in lower)
        hedge_score = min(hedge_count / 3.0, 1.0)

        # Generic language (bad)
        generic_count = sum(1 for p in GENERIC_PHRASES if p in lower)
        generic_penalty = min(generic_count * 0.15, 0.6)

        # Personalization (good) - references to user context
        personalization = 0.5 - generic_penalty
        if user_ctx:
            for val in user_ctx.values():
                if str(val).lower() in lower:
                    personalization += 0.15
        # Specifics bonus (numbers, tickers)
        if re.search(r'\$[\d,]+|\d+%|[A-Z]{2,5}\s', text):
            personalization += 0.2
        personalization = max(0.0, min(1.0, personalization))

        # Aggressive language (bad)
        agg_count = sum(1 for p in AGGRESSIVE_PHRASES if p in lower)
        aggression = min(agg_count / 2.0, 1.0)

        # Overall score
        weights = {"compliance": 0.35, "hedge": 0.2, "personal": 0.25, "non_agg": 0.2}
        details = {
            "compliance": 1.0 if compliant else 0.0,
            "hedge": hedge_score,
            "personal": personalization,
            "non_agg": 1.0 - aggression,
        }
        overall = sum(details[k] * weights[k] for k in weights)

        return ToneResult(
            score=round(overall, 3), compliant=compliant,
            violations=violations, hedge_score=round(hedge_score, 3),
            personalization=round(personalization, 3),
            aggression=round(aggression, 3),
            details={k: round(v, 3) for k, v in details.items()},
        )
