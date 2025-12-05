from dataclasses import dataclass
from .emotion import PAD

@dataclass
class Style:
    warmth: float; formality: float; hedging: float; emoji: float; pace: float

def style_from_pad(pad: PAD) -> Style:
    clamp = lambda x: max(0.0,min(1.0,x))
    warmth = clamp(0.5 + 0.4*pad.pleasure - 0.2*pad.dominance)
    formality = clamp(0.5 + 0.3*pad.dominance - 0.2*pad.arousal)
    hedging = clamp(0.6 - 0.4*pad.dominance + 0.1*(1-abs(pad.pleasure)))
    emoji = clamp(0.3 + 0.5*pad.pleasure)
    pace = clamp(0.5 + 0.4*pad.arousal)
    return Style(warmth, formality, hedging, emoji, pace)
