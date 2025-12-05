# src/affectgpt/policy.py
from __future__ import annotations
from .emotion import PAD
from .style import Style
from .llm import chat_llm

def render_response(
    user_text: str,
    pad: PAD,
    style: Style,
    strategy: str = "regulate",
    persona: dict | None = None,
) -> str:
    """
    Generate an emotionally adaptive reply via local LLM (Ollama).
    - pad: the *biased* PAD state (after persona bias)
    - style: Style dataclass (warmth/formality/hedging/emoji/pace)
    - strategy: "mirror" or "regulate" (nudge toward calm/focus)
    - persona: optional dict of sliders {valence_bias, arousal_bias, dominance_bias, weight, formality, directness, emoji}
    """
    persona = persona or {}

    system_prompt = (
        "You are AffectGPT, an empathetic, emotionally intelligent AI assistant. "
        "Respond succinctly (1–3 sentences), clearly, and supportively. Offer a small concrete next step or a focused question when helpful.\n\n"
        f"PAD mood (after persona bias): pleasure={pad.pleasure:.2f}, arousal={pad.arousal:.2f}, dominance={pad.dominance:.2f}.\n"
        f"Style knobs: warmth={style.warmth:.2f}, formality={style.formality:.2f}, "
        f"hedging={style.hedging:.2f}, emoji={style.emoji:.2f}, pace={style.pace:.2f}.\n"
        f"Strategy: {strategy} (mirror=reflect the user's tone; regulate=gently lower arousal / increase clarity and agency).\n"
        "Communication guidance:\n"
        "- Use a tone consistent with the PAD mood and style knobs.\n"
        "- If emoji>0.5, include at most ONE light emoji; otherwise avoid emojis.\n"
        "- Higher formality → more precise, professional wording.\n"
        "- Lower hedging → fewer 'maybe/might/perhaps'; higher hedging → softer language.\n"
        "- Keep answers concrete, avoid generic platitudes.\n"
    )

    # Persona context (helps the model align with the sliders' intent)
    system_prompt += (
        "\nPersona:\n"
        f"- valence_bias={persona.get('valence_bias', 0):.2f}, "
        f"arousal_bias={persona.get('arousal_bias', 0):.2f}, "
        f"dominance_bias={persona.get('dominance_bias', 0):.2f}, "
        f"weight={persona.get('weight', 0):.2f}\n"
        f"- directness={persona.get('directness', 0):.2f} (1=very direct, 0=very hedged), "
        f"formality={persona.get('formality', 0):.2f}, "
        f"emoji_playfulness={persona.get('emoji', 0):.2f}\n"
        "- Honor persona settings when wording the reply.\n"
    )

    try:
        return chat_llm(system_prompt, user_text)
    except Exception as e:
        # Graceful fallback if Ollama/server is unavailable
        return f"(Fallback) I’m having trouble reaching the local model: {e}"
