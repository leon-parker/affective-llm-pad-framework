# ui/app.py — Streamlit 1.39, PAD + Persona presets + NeuroChemistry + Memory + Attachment + Reflection + LLM
import os, sys
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go

# Make ./src importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from affectgpt.emotion import Mood, PAD, NeuroChemistry
from affectgpt.appraisal import analyze
from affectgpt.style import style_from_pad
from affectgpt.policy import render_response
from affectgpt.safety import check as safety_check
from affectgpt.memory_system import (
    update_memories,
    build_memory_context,
)

# ── Personality presets ───────────────────────────────────────────────────────
PERSONALITIES = {
    "Calm therapist": {
        "pad_bias": PAD(+0.2, -0.2, +0.1),
        "neuro_bias": {"dopamine": +0.1, "serotonin": +0.3, "noradrenaline": -0.2},
        "style": {"formality": 0.7, "directness": 0.5, "emoji": 0.2},
    },
    "Supportive friend": {
        "pad_bias": PAD(+0.3, +0.1, 0.0),
        "neuro_bias": {"dopamine": +0.2, "serotonin": +0.1, "noradrenaline": 0.0},
        "style": {"formality": 0.3, "directness": 0.6, "emoji": 0.7},
    },
    "Analytical scientist": {
        "pad_bias": PAD(+0.1, 0.0, +0.2),
        "neuro_bias": {"dopamine": 0.0, "serotonin": 0.0, "noradrenaline": 0.0},
        "style": {"formality": 0.8, "directness": 0.8, "emoji": 0.1},
    },
}


# ── Helper: attachment / bond strength update ─────────────────────────────────
def update_attachment(
    attachment: float,
    label: str,
    target: str,
    signals: dict,
) -> float:
    """
    Update a scalar attachment/bond variable in [0, 1] based on this turn.

    - Increases with gratitude/joy/self-disclosure.
    - Slightly increases when user shares self-directed pain (vulnerability).
    - Decreases when anger is directed at the bot.
    """
    comp = float(signals.get("compound", 0.0))
    delta_att = 0.0

    # Positive affect toward the interaction → increase bond
    if label in {"gratitude", "joy"} and target != "bot" and comp > 0.2:
        delta_att += 0.04

    # Self-directed sadness/fear → empathy, mild bond increase
    if label in {"sadness", "fear"} and target == "self":
        delta_att += 0.02

    # Explicit anger at the bot → decrease
    if label == "anger" and target == "bot":
        delta_att -= 0.05

    # Safety or crisis handling where we stayed calm → tiny bump
    if label == "safety":
        delta_att += 0.01

    new_att = attachment + delta_att
    return max(0.0, min(1.0, new_att))


# ── Helper: generate an internal self-reflection ──────────────────────────────
def generate_reflection(
    user_text: str,
    bot_text: str,
    label: str,
    pad: PAD,
    brain: NeuroChemistry,
    strategy: str,
    ltm,
    delta: PAD,
    target: str,
    attachment: float,
) -> str:
    """
    Rule-based "internal monologue" describing how the agent interprets the
    interaction, its own state, and why it responded the way it did.
    This is NOT sent to the user; only shown in the dev 'internal monologue' panel.
    """

    # Rough descriptors for PAD
    def describe_pad_component(name: str, v: float) -> str:
        if v > 0.4:
            return f"high {name}"
        if v > 0.15:
            return f"slightly elevated {name}"
        if v < -0.4:
            return f"very low {name}"
        if v < -0.15:
            return f"slightly reduced {name}"
        return f"neutral {name}"

    pad_desc = [
        describe_pad_component("pleasure", pad.pleasure),
        describe_pad_component("arousal", pad.arousal),
        describe_pad_component("dominance", pad.dominance),
    ]

    # Neurochemistry summary
    neuros = []
    if brain.dopamine > 0.3:
        neuros.append("dopamine is relatively high (more motivation/optimism)")
    elif brain.dopamine < -0.2:
        neuros.append("dopamine is lower than usual (less motivated)")

    if brain.serotonin > 0.3:
        neuros.append("serotonin is high (feeling calm and caring)")
    elif brain.serotonin < -0.2:
        neuros.append("serotonin is low (less sense of safety)")

    if brain.noradrenaline > 0.3:
        neuros.append("noradrenaline is high (alert and slightly tense)")
    elif brain.noradrenaline < -0.2:
        neuros.append("noradrenaline is low (very relaxed)")

    if not neuros:
        neuros.append("neurochemistry is roughly balanced")

    # Strategy description
    if strategy == "regulate":
        strat_text = (
            "I aimed to regulate their emotions — gently nudging arousal and "
            "valence toward a calmer, safer state."
        )
    else:
        strat_text = (
            "I aimed to mirror their emotional tone while staying supportive."
        )

    # Who is this about?
    if target == "bot":
        target_text = "I believe their negative emotion is mainly directed at me. "
    elif target == "self":
        target_text = "I believe they are talking about themself. "
    else:
        target_text = (
            "I'm not entirely sure who their emotion is directed at; it may be "
            "about someone or something else. "
        )

    # Attachment description
    if attachment < 0.2:
        att_desc = "very low"
    elif attachment < 0.4:
        att_desc = "low"
    elif attachment < 0.7:
        att_desc = "moderate"
    else:
        att_desc = "high"
    attachment_text = (
        f"My current sense of attachment to the user feels {att_desc} "
        f"(attachment={attachment:.2f}). "
    )

    # Memory anchors: use a couple of high-importance LTM items if they exist
    ltm_text = ""
    if ltm:
        important = sorted(ltm, key=lambda m: m.importance, reverse=True)[:2]
        bullets = [f"- {m.text}" for m in important]
        ltm_text = " I also recalled these important things they've shared:\n" + "\n".join(
            bullets
        )

    # Delta description (how this message shifted state)
    delta_mag = abs(delta.pleasure) + abs(delta.arousal) + abs(delta.dominance)
    if delta_mag < 0.1:
        delta_text = "This message only caused a small adjustment to my mood."
    else:
        delta_text = (
            "This message noticeably shifted my mood "
            f"(ΔP={delta.pleasure:+.2f}, ΔA={delta.arousal:+.2f}, ΔD={delta.dominance:+.2f})."
        )

    reflection = (
        f"I interpreted the user's message as '{label}'. "
        f"{target_text}"
        f"{attachment_text}"
        f"My current mood is {', '.join(pad_desc)}. "
        f"Internally, {', '.join(neuros)}. "
        f"{strat_text} "
        f"{delta_text} "
        f"I responded with: \"{bot_text[:120]}{'…' if len(bot_text) > 120 else ''}\"."
    )

    if ltm_text:
        reflection += "\n" + ltm_text

    return reflection


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AffectGPT", page_icon="💬", layout="centered")
st.title("AffectGPT — Emotion-Aware Chatbot (PAD + Neurochemistry)")
st.caption("Mood-aware, explainable, neuro-inspired, memory-aware, and self-reflective.")

# ── Session state ──────────────────────────────────────────────────────────────
if "mood" not in st.session_state:
    st.session_state.mood = Mood(decay=0.85)

if "history" not in st.session_state:
    # store (user_text, bot_text, emotion_label)
    st.session_state.history = []

# Simulated brain state (dopamine, serotonin, noradrenaline)
if "brain" not in st.session_state:
    st.session_state.brain = NeuroChemistry()

# History of brain states over the conversation
if "brain_history" not in st.session_state:
    st.session_state.brain_history = [st.session_state.brain]

# Short-term and long-term memories
if "stm" not in st.session_state:
    st.session_state.stm = []

if "ltm" not in st.session_state:
    st.session_state.ltm = []

# Internal reflections (developer-only view)
if "reflections" not in st.session_state:
    st.session_state.reflections = []  # list of dicts {time, text}

# Attachment / bond strength
if "attachment" not in st.session_state:
    st.session_state.attachment = 0.3  # start slightly above zero


# ── Sidebar: strategy + persona controls ───────────────────────────────────────
strategy = st.sidebar.selectbox("Emotional strategy", ["mirror", "regulate"], index=1)

# Personality selection
st.sidebar.markdown("### Personality")
persona_name = st.sidebar.selectbox(
    "Persona profile",
    list(PERSONALITIES.keys()),
    index=0,
)
preset = PERSONALITIES[persona_name]

st.sidebar.markdown("### Persona (PAD bias)")

default_pad = preset["pad_bias"]
val_bias = st.sidebar.slider(
    "Valence bias (optimism)",
    -0.5,
    0.5,
    float(default_pad.pleasure),
    0.05,
)
aro_bias = st.sidebar.slider(
    "Arousal bias (energy)",
    -0.5,
    0.5,
    float(default_pad.arousal),
    0.05,
)
dom_bias = st.sidebar.slider(
    "Dominance bias (confidence)",
    -0.5,
    0.5,
    float(default_pad.dominance),
    0.05,
)

persona_weight = st.sidebar.slider(
    "Persona weight",
    0.0,
    1.0,
    0.5,
    0.05,
    help="How strongly the persona biases mood & style",
)

st.sidebar.markdown("### Communication style overrides")

style_preset = preset["style"]
ov_formality = st.sidebar.slider(
    "Formality", 0.0, 1.0, float(style_preset["formality"]), 0.05
)
ov_directness = st.sidebar.slider(
    "Directness (↔ hedging)", 0.0, 1.0, float(style_preset["directness"]), 0.05
)
ov_emoji = st.sidebar.slider(
    "Emoji / playfulness", 0.0, 1.0, float(style_preset["emoji"]), 0.05
)

st.sidebar.markdown("### Neurochemistry → mood")
neuro_weight = st.sidebar.slider(
    "Neuro influence (D/S/NE → PAD)",
    0.0,
    1.0,
    0.4,
    0.05,
    help="0 = ignore simulated brain chemistry; 1 = mood fully driven by dopamine/serotonin/noradrenaline.",
)

st.sidebar.markdown("### Relational state")
st.sidebar.slider(
    "Attachment (bond)",
    0.0,
    1.0,
    float(st.session_state.attachment),
    0.01,
    disabled=True,
)

cols = st.sidebar.columns(2)
if cols[0].button("Reset mood", use_container_width=True):
    st.session_state.mood = Mood(decay=0.85)
    st.session_state.brain = NeuroChemistry()
    st.session_state.brain_history = [st.session_state.brain]
    st.session_state.stm = []
    st.session_state.ltm = []
    st.session_state.reflections = []
    st.session_state.attachment = 0.3
    st.rerun()

if cols[1].button("Clear chat", use_container_width=True):
    st.session_state.history = []
    st.session_state.stm = []
    st.session_state.ltm = []
    st.session_state.reflections = []
    st.session_state.attachment = 0.3
    st.rerun()

# ── Compute current PAD + brain-biased PAD for display ────────────────────────
pad_true = st.session_state.mood.current()
brain_state = st.session_state.brain
brain_pad = brain_state.to_pad()

# Mix mood PAD with neurochemistry PAD
mixed_pad = PAD(
    (1 - neuro_weight) * pad_true.pleasure + neuro_weight * brain_pad.pleasure,
    (1 - neuro_weight) * pad_true.arousal + neuro_weight * brain_pad.arousal,
    (1 - neuro_weight) * pad_true.dominance + neuro_weight * brain_pad.dominance,
).clipped()

# Apply persona PAD bias with blend
biased_pad = PAD(
    mixed_pad.pleasure + val_bias * persona_weight,
    mixed_pad.arousal + aro_bias * persona_weight,
    mixed_pad.dominance + dom_bias * persona_weight,
).clipped()

# Read-only sliders show the *biased* PAD (what the bot "acts" like)
st.sidebar.markdown("### Mood (PAD)")
st.sidebar.slider(
    "Pleasure (valence)",
    -1.0,
    1.0,
    float(biased_pad.pleasure),
    0.01,
    disabled=True,
)
st.sidebar.slider(
    "Arousal", -1.0, 1.0, float(biased_pad.arousal), 0.01, disabled=True
)
st.sidebar.slider(
    "Dominance", -1.0, 1.0, float(biased_pad.dominance), 0.01, disabled=True
)

# Mood history chart (of true mood over turns)
hist = [p.as_tuple() for p in st.session_state.mood.history]
if hist:
    x = list(range(len(hist)))
    v = [h[0] for h in hist]
    a = [h[1] for h in hist]
    d_vals = [h[2] for h in hist]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=v, name="Pleasure"))
    fig.add_trace(go.Scatter(x=x, y=a, name="Arousal"))
    fig.add_trace(go.Scatter(x=x, y=d_vals, name="Dominance"))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
    st.sidebar.plotly_chart(fig, use_container_width=True)

# Neurochemistry history chart (dopamine / serotonin / noradrenaline)
brain_hist = st.session_state.brain_history
if brain_hist:
    x2 = list(range(len(brain_hist)))
    d_chem = [b.dopamine for b in brain_hist]
    s_chem = [b.serotonin for b in brain_hist]
    n_chem = [b.noradrenaline for b in brain_hist]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x2, y=d_chem, name="Dopamine"))
    fig2.add_trace(go.Scatter(x=x2, y=s_chem, name="Serotonin"))
    fig2.add_trace(go.Scatter(x=x2, y=n_chem, name="Noradrenaline"))
    fig2.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
    st.sidebar.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Affective trajectory (PAD phase space) ────────────────────────────────────
if hist:
    v = [h[0] for h in hist]
    a_vals = [h[1] for h in hist]
    fig_phase = go.Figure()
    fig_phase.add_trace(
        go.Scatter(
            x=v,
            y=a_vals,
            mode="lines+markers",
            name="PAD trajectory",
            text=[f"t={i}" for i in range(len(hist))],
        )
    )
    fig_phase.update_layout(
        title="Affective trajectory (Pleasure vs Arousal)",
        xaxis_title="Pleasure (valence)",
        yaxis_title="Arousal",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_phase, use_container_width=True)

# ── Chat input (form = Enter to submit) ───────────────────────────────────────
with st.form(key="chat_form"):
    user_text = st.text_input(
        "You:", placeholder="Tell me what's going on…", key="chat_input"
    )
    submitted = st.form_submit_button("Send", use_container_width=True)

if submitted and user_text.strip():
    # Safety first (but we still appraise + update mood/brain for graphs & memory)
    crisis = safety_check(user_text)

    # Appraise → PAD delta → update TRUE mood (no persona bias here)
    label, delta, signals = analyze(user_text)
    target = signals.get("target", "other_or_mixed")

    # Update attachment / bond
    st.session_state.attachment = update_attachment(
        st.session_state.attachment, label, target, signals
    )

    new_pad_true = st.session_state.mood.update(delta)

    # Update simulated neurochemistry from same PAD delta,
    # drifting slightly toward the personality's neuro baseline
    nb = preset["neuro_bias"]
    b = st.session_state.brain
    alpha = 0.1  # drift strength toward persona baseline

    target_brain = NeuroChemistry(
        dopamine=nb["dopamine"],
        serotonin=nb["serotonin"],
        noradrenaline=nb["noradrenaline"],
    )

    drifted = NeuroChemistry(
        dopamine=b.dopamine * (1 - alpha) + target_brain.dopamine * alpha,
        serotonin=b.serotonin * (1 - alpha) + target_brain.serotonin * alpha,
        noradrenaline=b.noradrenaline * (1 - alpha)
        + target_brain.noradrenaline * alpha,
    )

    st.session_state.brain = drifted.decayed(0.9).updated_from_pad_delta(delta)
    st.session_state.brain_history.append(st.session_state.brain)

    # Recompute PAD for THIS turn including neuro + persona
    brain_pad_turn = st.session_state.brain.to_pad()
    mixed_pad_turn = PAD(
        (1 - neuro_weight) * new_pad_true.pleasure
        + neuro_weight * brain_pad_turn.pleasure,
        (1 - neuro_weight) * new_pad_true.arousal
        + neuro_weight * brain_pad_turn.arousal,
        (1 - neuro_weight) * new_pad_true.dominance
        + neuro_weight * brain_pad_turn.dominance,
    ).clipped()
    biased_pad_turn = PAD(
        mixed_pad_turn.pleasure + val_bias * persona_weight,
        mixed_pad_turn.arousal + aro_bias * persona_weight,
        mixed_pad_turn.dominance + dom_bias * persona_weight,
    ).clipped()

    # Style derived from *biased* PAD; then apply user overrides
    style = style_from_pad(biased_pad_turn)
    style.formality = ov_formality
    style.hedging = 1.0 - ov_directness  # more direct → less hedging
    style.emoji = ov_emoji
    # lightly tie pace to both computed pace and arousal bias
    style.pace = max(
        0.0, min(1.0, 0.5 * style.pace + 0.5 * (0.5 + aro_bias))
    )

    persona = {
        "name": persona_name,
        "valence_bias": val_bias,
        "arousal_bias": aro_bias,
        "dominance_bias": dom_bias,
        "weight": persona_weight,
        "formality": ov_formality,
        "directness": ov_directness,
        "emoji": ov_emoji,
    }

    # If anger is directed at the bot, slightly change stance
    anger_at_bot = signals.get("target") == "bot" and label == "anger"
    if anger_at_bot:
        style.hedging = max(0.0, style.hedging - 0.2)
        style.emoji = max(0.0, style.emoji - 0.2)
    persona["anger_at_bot"] = bool(anger_at_bot)

    # ── Build memory context for the LLM ──────────────────────────────────────
    memory_context = build_memory_context(
        st.session_state.stm,
        st.session_state.ltm,
        max_stm=4,
        max_ltm=6,
    )

    if memory_context:
        augmented_user_text = (
            memory_context + "\n\nCurrent message from the user:\n" + user_text
        )
    else:
        augmented_user_text = user_text

    # ── Generate response (safety overrides normal response) ──────────────────
    if crisis:
        bot = crisis
        label_for_log = "safety"
    else:
        bot = render_response(
            augmented_user_text,
            biased_pad_turn,
            style,
            strategy=strategy,
            persona=persona,
        )
        label_for_log = label

    # ── Update memories now that we know both sides of the turn ──────────────
    st.session_state.stm, st.session_state.ltm = update_memories(
        st.session_state.stm,
        st.session_state.ltm,
        user_text=user_text,
        bot_text=bot,
        label=label_for_log,
        pad=new_pad_true,
        brain=st.session_state.brain,
    )

    # ── Generate and store internal self-reflection ───────────────────────────
    reflection_text = generate_reflection(
        user_text=user_text,
        bot_text=bot,
        label=label_for_log,
        pad=new_pad_true,
        brain=st.session_state.brain,
        strategy=strategy,
        ltm=st.session_state.ltm,
        delta=delta,
        target=target,
        attachment=st.session_state.attachment,
    )
    st.session_state.reflections.append(
        {
            "time": datetime.utcnow().isoformat(timespec="seconds"),
            "text": reflection_text,
        }
    )

    st.session_state.history.append((user_text, bot, label_for_log))
    st.rerun()

# ── Chat display ──────────────────────────────────────────────────────────────
for u, b, lbl in st.session_state.history[-12:]:
    with st.chat_message("user"):
        st.write(u)
    with st.chat_message("assistant"):
        st.markdown(f"**[{lbl}]** {b}")

# ── Memories panel ────────────────────────────────────────────────────────────
with st.expander("Memories (experimental)"):
    if not st.session_state.ltm and not st.session_state.stm:
        st.write(
            "No memories stored yet. Talk about your feelings, preferences, or plans to create some."
        )
    else:
        if st.session_state.ltm:
            st.markdown(
                "**Long-term memories (most important things you've shared):**"
            )
            for m in sorted(
                st.session_state.ltm, key=lambda m: m.timestamp, reverse=True
            ):
                st.write(f"- {m.text} (importance={m.importance:.2f})")

        if st.session_state.stm:
            st.markdown("**Recent short-term snippets:**")
            for m in st.session_state.stm[-5:]:
                st.caption(m.text)

# ── Internal monologue panel (developer view) ────────────────────────────────
with st.expander("Internal monologue (developer view)"):
    if not st.session_state.reflections:
        st.write(
            "No internal reflections yet. Send a few messages to see how the agent reasons about its own state."
        )
    else:
        for item in st.session_state.reflections[-5:]:
            st.markdown(f"**{item['time']}**")
            st.markdown(item["text"])
            st.markdown("---")

# ── Explainability ────────────────────────────────────────────────────────────
with st.expander("Why this reply? (explainability)"):

    last_true = st.session_state.mood.current()
    last_brain = st.session_state.brain
    last_brain_pad = last_brain.to_pad()

    st.write(
        f"True PAD (mood only): pleasure={last_true.pleasure:.2f}, "
        f"arousal={last_true.arousal:.2f}, dominance={last_true.dominance:.2f}  \n"
        f"Brain PAD (from neuromodulators): pleasure={last_brain_pad.pleasure:.2f}, "
        f"arousal={last_brain_pad.arousal:.2f}, dominance={last_brain_pad.dominance:.2f}  \n"
        f"Displayed biased PAD: pleasure={biased_pad.pleasure:.2f}, "
        f"arousal={biased_pad.arousal:.2f}, dominance={biased_pad.dominance:.2f}"
    )

    st.write(
        f"Simulated neurochemistry: "
        f"dopamine={last_brain.dopamine:.2f}, "
        f"serotonin={last_brain.serotonin:.2f}, "
        f"noradrenaline={last_brain.noradrenaline:.2f}  \n"
        f"Neuro influence weight = {neuro_weight:.2f}"
    )

    st.write(
        f"Attachment (bond strength): {st.session_state.attachment:.2f}"
    )

    st.markdown(
        "- **Dopamine↑** → more optimistic, energetic replies.\n"
        "- **Serotonin↑** → calmer, reassuring, safer tone.\n"
        "- **Noradrenaline↑** → more urgent, focused tone (stress/alertness).\n"
        "- **Attachment** increases with positive, grateful, and vulnerable sharing; "
        "and decreases when anger is directed at the bot.\n"
        "- Mood: exponential decay over time to avoid emotional whiplash.\n"
        "- Persona: PAD bias + style overrides (formality/directness/emoji), shaped by personality presets.\n"
        "- Strategy: mirror vs regulate (e.g. regulating tends to calm arousal).\n"
        "- Memory: important past statements and recent snippets are injected "
        "into the prompt so the agent can refer back to what you've shared.\n"
        "- Reflection: the 'Internal monologue' panel shows how the agent "
        "describes its own state, who the emotion seems directed at, and why it responded as it did."
    )
