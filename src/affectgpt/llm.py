# src/affectgpt/llm.py
import os, requests

OLLAMA_URL   = os.getenv("OLLAMA_URL",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("LLM_MODEL",   "llama3.2:3b")

def chat_llm(system: str, user: str, temperature: float = 0.6, max_tokens: int = 350) -> str:
    prompt = f"{system}\n\nUser: {user}\nAssistant:"
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "options": {"temperature": temperature},
            "stream": False,
        },
        timeout=60,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()
