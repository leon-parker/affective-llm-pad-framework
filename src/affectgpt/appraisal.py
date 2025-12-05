import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .emotion import PAD

_v = SentimentIntensityAnalyzer()

EMOTION_TO_PAD = {
    "joy":       PAD(+0.8, +0.2, +0.2),
    "gratitude": PAD(+0.6, +0.1, +0.2),
    "anger":     PAD(-0.7, +0.6, +0.4),
    "sadness":   PAD(-0.7, -0.4, -0.4),
    "fear":      PAD(-0.8, +0.5, -0.6),
    "neutral":   PAD(0.0, 0.0, 0.0),
}

RULES = [
    (re.compile(r"\b(thanks|appreciate|grateful)\b", re.I), "gratitude"),
    (re.compile(r"\b(angry|furious|pissed)\b", re.I), "anger"),
    (re.compile(r"\b(sad|upset|depressed)\b", re.I), "sadness"),
    (re.compile(r"\b(scared|afraid|panic|anxious)\b", re.I), "fear"),
    (re.compile(r"\b(happy|great|awesome|love it)\b", re.I), "joy"),
]


def detect_target(text: str, compound: float) -> str:
    """
    Infer who negative emotion is about:
      - 'self'  → user is talking about themself (I, me, my…)
      - 'bot'   → user is talking to / about the assistant (you…)
      - 'other_or_mixed' otherwise

    Only used when sentiment is clearly negative.
    """
    t = text.lower()

    # If not clearly negative, don't over-interpret
    if compound > -0.3:
        return "other_or_mixed"

    has_first = bool(re.search(r"\b(i|me|my|mine|i'm|i am|i feel)\b", t))
    has_second = bool(re.search(r"\byou\b|\byou're\b|\byou are\b", t))

    if has_second and not has_first:
        return "bot"
    if has_first and not has_second:
        return "self"
    return "other_or_mixed"


def analyze(text: str):
    """
    Analyze a user utterance and return:
        (emotion_label, PAD_delta, signals_dict)

    signals_dict includes VADER scores plus a 'target' field indicating
    whether negativity seems self-directed, directed at the bot, or other.
    """
    t = text.strip()
    if not t:
        return "neutral", EMOTION_TO_PAD["neutral"], {
            "compound": 0.0,
            "target": "other_or_mixed",
        }

    # ── 1. Keyword rules (fast path, with target added to signals) ───────────
    for rx, label in RULES:
        if rx.search(t):
            scores = _v.polarity_scores(t)
            comp = scores.get("compound", 0.0)
            scores["target"] = detect_target(t, comp)
            scores["kw:" + label] = 1.0
            return label, EMOTION_TO_PAD[label], scores

    # ── 2. Fallback to VADER sentiment ───────────────────────────────────────
    scores = _v.polarity_scores(t)
    comp = scores.get("compound", 0.0)
    target = detect_target(t, comp)
    scores["target"] = target

    # Strongly positive
    if comp >= 0.4:
        label = "joy"
        return label, EMOTION_TO_PAD[label], scores

    # Strongly negative
    if comp <= -0.4:
        # If clearly self-directed → sadness/fear about self
        if target == "self":
            label = "sadness"
        # If clearly directed at the assistant → anger at bot
        elif target == "bot":
            label = "anger"
        else:
            # generic negative: lean sadness by default
            label = "sadness"
        return label, EMOTION_TO_PAD[label], scores

    # Mildly negative / mixed
    if comp < 0:
        # If user talks about themselves, lean sadness
        if target == "self":
            label = "sadness"
        # If user is mildly negative at the bot, treat as low-intensity anger
        elif target == "bot":
            label = "anger"
        else:
            label = "sadness"
        return label, EMOTION_TO_PAD[label], scores

    # Neutral-ish / mildly positive
    if comp > 0:
        label = "joy"
    else:
        label = "neutral"
    return label, EMOTION_TO_PAD[label], scores
