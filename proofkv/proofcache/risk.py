from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RiskDecision:
    action: str
    confidence: float
    threshold: float
    budget_multiplier: float
    fallback_to_full: bool


class CalibratedRiskController:
    """Finite-sample conservative thresholding for compact-route acceptance."""

    def __init__(
        self,
        target_risk: float = 0.02,
        min_coverage: float = 0.80,
        expansion_multipliers: Sequence[float] = (1.0, 2.0),
    ):
        if not 0.0 < target_risk < 1.0:
            raise ValueError("target_risk must lie in (0, 1)")
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must lie in [0, 1]")
        self.target_risk = target_risk
        self.min_coverage = min_coverage
        self.expansion_multipliers = tuple(expansion_multipliers)
        self.threshold = 1.0
        self.calibration_coverage = 0.0
        self.calibration_risk_upper = 1.0

    @staticmethod
    def conservative_risk(errors: int, accepted: int) -> float:
        # Add-one finite-sample upper proxy. It prevents a zero-error calibration set
        # from being treated as proof of zero deployment risk.
        return (errors + 1) / (accepted + 1) if accepted else 1.0

    def fit(self, confidences: Sequence[float], sufficient: Sequence[bool]) -> "CalibratedRiskController":
        if len(confidences) != len(sufficient) or not confidences:
            raise ValueError("non-empty calibration arrays must have equal length")
        candidates = sorted(set(float(value) for value in confidences))
        feasible: list[tuple[float, float, float]] = []
        for threshold in candidates:
            accepted = [idx for idx, value in enumerate(confidences) if value >= threshold]
            errors = sum(not sufficient[idx] for idx in accepted)
            risk_upper = self.conservative_risk(errors, len(accepted))
            coverage = len(accepted) / len(confidences)
            if risk_upper <= self.target_risk:
                feasible.append((coverage, -threshold, risk_upper))
        if feasible:
            coverage, negative_threshold, risk_upper = max(feasible)
            self.threshold = -negative_threshold
            self.calibration_coverage = coverage
            self.calibration_risk_upper = risk_upper
        else:
            self.threshold = 1.0 + 1e-9
            self.calibration_coverage = 0.0
            self.calibration_risk_upper = 1.0
        return self

    @property
    def gate_passed(self) -> bool:
        return self.calibration_coverage >= self.min_coverage and self.calibration_risk_upper <= self.target_risk

    def decide(self, confidence: float) -> RiskDecision:
        if confidence >= self.threshold:
            return RiskDecision("compact", confidence, self.threshold, self.expansion_multipliers[0], False)
        if len(self.expansion_multipliers) > 1 and confidence >= 0.5 * self.threshold:
            return RiskDecision("expand", confidence, self.threshold, self.expansion_multipliers[-1], False)
        return RiskDecision("full_fallback", confidence, self.threshold, float("inf"), True)
