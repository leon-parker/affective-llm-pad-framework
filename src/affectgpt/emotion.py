from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class PAD:
    """Represents the Pleasure–Arousal–Dominance (PAD) affective state."""
    pleasure: float
    arousal: float
    dominance: float

    def clipped(self) -> "PAD":
        """Clip PAD values to [-1, +1] to prevent overflow."""
        return PAD(
            max(-1.0, min(1.0, self.pleasure)),
            max(-1.0, min(1.0, self.arousal)),
            max(-1.0, min(1.0, self.dominance)),
        )

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return (pleasure, arousal, dominance) as a tuple."""
        return (self.pleasure, self.arousal, self.dominance)

    @staticmethod
    def zero() -> "PAD":
        """Return a neutral (0,0,0) PAD state."""
        return PAD(0.0, 0.0, 0.0)

    def scale(self, factor: float) -> "PAD":
        """Scale all PAD dimensions by a factor."""
        return PAD(
            self.pleasure * factor,
            self.arousal * factor,
            self.dominance * factor,
        )


class Mood:
    """Maintains a decaying emotional mood state in PAD space."""

    def __init__(self, decay: float = 0.85, baseline: Optional[PAD] = None):
        """
        Args:
            decay: how quickly past emotions fade (0.8 = slow, 0.95 = very persistent)
            baseline: starting PAD state (defaults to neutral)
        """
        self.decay = decay
        self.state = baseline if baseline else PAD.zero()
        self.history: List[PAD] = [self.state]

    def update(self, delta: PAD, blend: float = 0.25) -> PAD:
        """Update the mood based on a new PAD delta and store history."""
        decayed = self.state.scale(self.decay)
        self.state = PAD(
            decayed.pleasure + delta.pleasure * blend,
            decayed.arousal + delta.arousal * blend,
            decayed.dominance + delta.dominance * blend,
        ).clipped()
        self.history.append(self.state)
        return self.state

    def current(self) -> PAD:
        """Return the current PAD mood state."""
        return self.state

    def reset(self, baseline: Optional[PAD] = None):
        """Reset mood back to neutral or to a custom baseline."""
        self.state = baseline if baseline else PAD.zero()
        self.history = [self.state]


# ── Computational neurochemistry model ────────────────────────────────────────

@dataclass
class NeuroChemistry:
    """
    Simple computational model of three neuromodulators.

    Values are in [-1, +1]:
    - dopamine: reward / motivation
    - serotonin: safety / well-being
    - noradrenaline: alertness / stress
    """
    dopamine: float = 0.0
    serotonin: float = 0.0
    noradrenaline: float = 0.0

    def clipped(self) -> "NeuroChemistry":
        """Clamp all values to [-1, +1]."""
        def clip(x: float) -> float:
            return max(-1.0, min(1.0, x))

        return NeuroChemistry(
            clip(self.dopamine),
            clip(self.serotonin),
            clip(self.noradrenaline),
        )

    def to_pad(self) -> PAD:
        """
        Map neuromodulator levels to a PAD *bias*.

        Intuition:
        - Dopamine → positive valence & moderate arousal.
        - Serotonin → positive valence, slightly calmer & less dominant.
        - Noradrenaline → high arousal & dominance, slightly lower valence.
        """
        d = self.dopamine
        s = self.serotonin
        n = self.noradrenaline

        pleasure = 0.5 * d + 0.6 * s - 0.2 * n
        arousal = 0.4 * d - 0.2 * s + 0.7 * n
        dominance = 0.3 * d - 0.3 * s + 0.6 * n

        return PAD(pleasure, arousal, dominance).clipped()

    def decayed(self, factor: float = 0.9) -> "NeuroChemistry":
        """Exponential decay back toward neutral (0,0,0)."""
        return NeuroChemistry(
            self.dopamine * factor,
            self.serotonin * factor,
            self.noradrenaline * factor,
        )

    def updated_from_pad_delta(self, delta: PAD, learning_rate: float = 0.4) -> "NeuroChemistry":
        """
        Update neuromodulators in response to a new PAD delta.

        - Positive pleasure boosts dopamine & serotonin.
        - Negative pleasure + high arousal boosts noradrenaline (stress).
        - Dominance nudges dopamine slightly (sense of agency).
        """
        d = self.dopamine
        s = self.serotonin
        n = self.noradrenaline

        # Reward / safety: positive valence
        if delta.pleasure >= 0:
            d += learning_rate * delta.pleasure
            s += 0.5 * learning_rate * delta.pleasure
        else:
            # Negative valence → threat, more NE
            n += -0.4 * learning_rate * delta.pleasure  # delta.pleasure is negative

        # High arousal → more NE
        if delta.arousal > 0:
            n += 0.6 * learning_rate * delta.arousal

        # Sense of control nudges dopamine
        d += 0.1 * learning_rate * delta.dominance

        return NeuroChemistry(d, s, n).clipped()
