"""One wrapper for any LLM API (chat + embeddings). Change .env, not code."""
import os, json, re, time
from dotenv import load_dotenv

load_dotenv()
PROVIDER = os.getenv("PROVIDER", "openai").lower()
MODEL = os.getenv("MODEL", "gpt-4o-mini")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL") or None
EMBED_MODEL = os.getenv("EMBED_MODEL", "")

_client = None
_embed_client = None


def _get_client():
    global _client
    if _client is None:
        if PROVIDER == "anthropic":
            import anthropic
            _client = anthropic.Anthropic(api_key=API_KEY)
        else:
            from openai import OpenAI
            _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def chat(prompt, system="You are a helpful assistant.", history=None,
         temperature=0.2, max_tokens=1500, retries=3):
    """history = [{"role":"user"|"assistant","content":"..."}]"""
    msgs = (history or []) + [{"role": "user", "content": prompt}]
    c = _get_client()
    for attempt in range(retries):
        try:
            if PROVIDER == "anthropic":
                r = c.messages.create(model=MODEL, system=system, messages=msgs,
                                      temperature=temperature, max_tokens=max_tokens)
                return r.content[0].text
            r = c.chat.completions.create(
                model=MODEL, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}] + msgs)
            return r.choices[0].message.content
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def ask_json(prompt, system="Return ONLY valid JSON. No prose, no markdown fences."):
    """Structured output. Describe the JSON shape in the prompt."""
    for _ in range(3):
        raw = chat(prompt, system=system, temperature=0)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            prompt += "\n\nYour last reply was not valid JSON. Return only JSON."
    raise ValueError("Model did not return valid JSON")


def embeddings_available():
    return bool(EMBED_MODEL)


def embed(texts, batch=64, retries=3):
    """Text -> list of vectors (OpenAI-compatible embeddings endpoint)."""
    global _embed_client
    if not EMBED_MODEL:
        raise RuntimeError("EMBED_MODEL not set")
    if _embed_client is None:
        from openai import OpenAI
        _embed_client = OpenAI(api_key=os.getenv("EMBED_API_KEY") or API_KEY,
                               base_url=os.getenv("EMBED_BASE_URL") or BASE_URL)
    out = []
    for i in range(0, len(texts), batch):
        part = [t[:6000] or " " for t in texts[i:i + batch]]
        for attempt in range(retries):
            try:
                r = _embed_client.embeddings.create(model=EMBED_MODEL, input=part)
                out += [d.embedding for d in r.data]
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
    return out


if __name__ == "__main__":  # smoke test: python -m src.llm
    print("CHAT:", chat("Say 'API works' in 3 words."))
    if embeddings_available():
        print("EMBED dims:", len(embed(["hello"])[0]))
