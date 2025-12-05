📘 README.md (Replace your current one with this)
# Neuro-modulation  
*A neuro-inspired, emotionally adaptive chatbot using PAD affect modelling, simulated neuromodulators, memory systems and self-reflection.*

Neuro-modulation is an experimental conversational agent designed to explore whether a chatbot can behave with more realistic affective dynamics. It combines several computational-neuroscience–inspired mechanisms:

- **PAD Emotion Model** — the agent maintains a continuous Pleasure–Arousal–Dominance mood state.
- **Simulated Neurochemistry** — dopamine, serotonin and noradrenaline dynamically influence mood and communication style.
- **Affective Appraisal** — messages are interpreted as joy, sadness, anger, fear, etc., shifting mood accordingly.
- **Persona Modulation** — sliders let you bias the agent toward optimism, energy or confidence.
- **Memory System** — the agent stores short-term and long-term memories of the conversation to build stable preferences and context.
- **Attachment Style** — alters how the system regulates or mirrors emotional tone.
- **Internal Self-Reflection** — after each turn, the agent generates a private monologue describing its emotional state, reasoning and memory retrieval.

The system runs as a **Streamlit app** and communicates with an LLM through a policy module that blends mood, persona, memory and safety behaviours.

This project aims to demonstrate how computational neuroscience concepts can be used to shape more lifelike, adaptive conversational behaviour.

---

## 🚀 Running the App

Inside your project folder:

```bash
pip install -r requirements.txt
streamlit run ui/app.py

📂 Structure
ui/app.py                – Main Streamlit interface
src/affectgpt/emotion.py – PAD model + neurochemistry
src/affectgpt/appraisal.py – Affective appraisal + target detection
src/affectgpt/style.py   – Style modulation
src/affectgpt/policy.py  – LLM response generation
src/affectgpt/memory_system.py – STM + LTM mechanism

🧠 Vision

Neuro-modulation is a step toward emotionally coherent agents that:

maintain internal states

adapt behaviour over time

form stable preferences

regulate or mirror human emotion

track their own internal narrative

It is not meant to imitate sentience — but to explore the mechanisms underlying affective behaviour in computational systems.

📜 License

MIT License.


---

# Want me to:
✅ push the updated README to GitHub for you (I’ll give you the commands)?  
✅ add a project logo?  
✅ generate an API for external apps to talk to your model?  
Just tell me and I’ll do it.