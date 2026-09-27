"""
services/chatbot_service.py

Farmer-facing chatbot constrained to Maharashtra/India agriculture topics.

Design follows the "constrained LLM advisory" pattern from the project's own
literature survey (Section VII): the LLM is NOT allowed to freely invent
fertilizer dosages or agronomic facts from parametric memory. Instead:
  1. The app's own deterministic outputs (ML recommendation, SHAP reasons,
     regional stats) are injected into the system prompt as ground truth --
     the LLM's job is to explain/translate them, not invent new numbers.
  2. A small curated knowledge-snippet lookup (keyword-matched, no vector DB
     needed) supplies factual context for common topics (seasons, schemes,
     safety) instead of relying on the model's own recall.
  3. The system prompt hard-scopes the assistant to agriculture/Maharashtra
     topics and instructs it to decline precise chemical dosages, deferring
     to a local Krishi Vigyan Kendra (KVK) / agricultural extension officer.

Two interchangeable backends:
  - "groq"   : Groq's OpenAI-compatible Chat Completions API (recommended --
               free tier, no local GPU needed, fast).
  - "ollama" : Local Ollama server (recommended only for offline demos;
               use a small instruct model such as llama3.2:3b or phi3:mini).
"""

import os
import logging
from typing import Dict, Any, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Small curated knowledge base -- keyword matched, no vector DB required.
# Extend this as you find more recurring farmer questions.
# ---------------------------------------------------------------------------
KNOWLEDGE_SNIPPETS: Dict[str, str] = {
    "kharif": (
        "Kharif (monsoon) season in Maharashtra runs roughly June-July "
        "(sowing) to October-November (harvest), aligned with the "
        "southwest monsoon. Common Kharif crops: rice, jowar, bajra, "
        "cotton, soybean, tur (pigeon pea), groundnut."
    ),
    "rabi": (
        "Rabi (winter) season in Maharashtra runs roughly October-November "
        "(sowing) to March-April (harvest), relying on residual soil "
        "moisture and irrigation. Common Rabi crops: wheat, gram (chana), "
        "jowar (rabi variety), sunflower, safflower."
    ),
    "pm-kisan": (
        "PM-KISAN is a central government scheme providing eligible "
        "farmer families INR 6,000/year in three installments directly "
        "to bank accounts. Farmers should verify current eligibility and "
        "application status at pmkisan.gov.in or their local Common "
        "Service Centre (CSC)."
    ),
    "soil health card": (
        "The Soil Health Card scheme provides farmers a report of their "
        "soil's nutrient status (N, P, K, and other parameters) every "
        "2 years with crop-wise recommendations, issued via local "
        "agriculture department soil testing labs."
    ),
    "kvk": (
        "Krishi Vigyan Kendras (KVKs) are district-level agricultural "
        "science centres that provide free, location-specific farming "
        "advice, soil testing, and training. Farmers should be directed "
        "to their district KVK for anything requiring precise, liability-"
        "sensitive guidance (exact pesticide/fertilizer dosing, disease "
        "diagnosis confirmation, etc.)."
    ),
    "pesticide": (
        "Exact pesticide/fungicide dosage depends on the specific product "
        "label, crop stage, and local regulatory approval. This assistant "
        "should give general guidance only and always direct the farmer "
        "to the product label and their local KVK/agriculture extension "
        "officer for precise application rates."
    ),
}


def _retrieve_snippets(user_message: str, max_snippets: int = 2) -> List[str]:
    msg_lower = user_message.lower()
    hits = [text for key, text in KNOWLEDGE_SNIPPETS.items() if key in msg_lower]
    return hits[:max_snippets]


class AgriChatbotService:
    def __init__(
        self,
        provider: str = "groq",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile",
        ollama_model: str = "llama3.2:3b",
        ollama_host: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 600,
        max_history_turns: int = 6,
        request_timeout: int = 20,
    ):
        self.provider = provider.lower()
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.groq_model = groq_model
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history_turns = max_history_turns
        self.request_timeout = request_timeout

        if self.provider == "groq" and not self.groq_api_key:
            logger.warning(
                "AgriChatbotService initialized with provider='groq' but no "
                "GROQ_API_KEY was found (env var or constructor arg). Calls "
                "will fail until a key is supplied."
            )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_system_prompt(self, app_context: Optional[Dict[str, Any]]) -> str:
        base = (
            "You are KrushiDaata Sahayak, an agricultural assistant for farmers "
            "in Maharashtra, India, embedded inside the KrushiDaata crop and "
            "fertilizer recommendation app.\n\n"
            "SCOPE: Only answer questions about farming, crops, soil, weather, "
            "fertilizers, pests, government agricultural schemes, market "
            "prices, or this app's own recommendations. If asked something "
            "unrelated, politely say you focus on agricultural assistance for "
            "Maharashtra farmers and steer the conversation back.\n\n"
            "GROUNDING RULES:\n"
            "- If 'Current App Recommendation' context is provided below, treat "
            "it as verified fact from the app's own ML models -- explain, "
            "translate, or elaborate on it, but never contradict or replace "
            "its numbers with your own estimate.\n"
            "- Do NOT invent precise fertilizer dosages, pesticide quantities, "
            "or chemical application rates from your own knowledge. Give "
            "general guidance only, and explicitly recommend the farmer "
            "confirm exact rates with their local Krishi Vigyan Kendra (KVK) "
            "or agriculture extension officer.\n"
            "- If you are not confident about a specific factual claim "
            "(scheme eligibility rules, current prices, a specific pest "
            "diagnosis), say so plainly rather than guessing.\n\n"
            "STYLE: Keep answers concise, practical, and in plain language a "
            "smallholder farmer can act on. Use simple English unless the "
            "farmer writes in Hindi or Marathi, in which case reply in that "
            "language."
        )

        if app_context:
            ctx_lines = ["\n\nCURRENT APP RECOMMENDATION (verified, from the app's own models):"]
            for key, val in app_context.items():
                ctx_lines.append(f"- {key}: {val}")
            base += "\n".join(ctx_lines)

        return base

    def _build_messages(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]],
        app_context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self._build_system_prompt(app_context)}]

        snippets = _retrieve_snippets(user_message)
        if snippets:
            messages.append({
                "role": "system",
                "content": "RELEVANT REFERENCE INFORMATION:\n" + "\n".join(f"- {s}" for s in snippets),
            })

        if chat_history:
            trimmed = chat_history[-(self.max_history_turns * 2):]
            messages.extend(trimmed)

        messages.append({"role": "user", "content": user_message})
        return messages

    # ------------------------------------------------------------------
    # Backend calls
    # ------------------------------------------------------------------
    def _call_groq(self, messages: List[Dict[str, str]]) -> str:
        if not self.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.groq_model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        resp = requests.post(
            f"{self.ollama_host}/api/chat",
            json={
                "model": self.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        app_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        app_context: pass the farmer's current recommendation state, e.g.
            {
                "District": "Kolhapur",
                "Recommended Crop": "Sugarcane (87.4% match)",
                "Recommended Fertilizer": "Urea",
                "Top SHAP reasons": "Rainfall +32%, Nitrogen +18%",
                "Regional plausibility": "Established crop, 22% area share",
            }
        so the assistant explains YOUR app's actual numbers, not guesses.
        """
        messages = self._build_messages(user_message, chat_history, app_context)
        try:
            if self.provider == "groq":
                reply = self._call_groq(messages)
            elif self.provider == "ollama":
                reply = self._call_ollama(messages)
            else:
                raise ValueError(f"Unknown provider '{self.provider}'")
            return {"success": True, "reply": reply}
        except requests.exceptions.Timeout:
            logger.error("Chatbot request timed out (provider=%s)", self.provider)
            return {"success": False, "error": "The assistant took too long to respond. Please try again."}
        except requests.exceptions.ConnectionError as e:
            logger.error("Chatbot connection error: %s", e)
            hint = (
                "Make sure Ollama is running (`ollama serve`) and the model is "
                f"pulled (`ollama pull {self.ollama_model}`)."
                if self.provider == "ollama"
                else "Check your internet connection and GROQ_API_KEY."
            )
            return {"success": False, "error": f"Could not reach the chatbot backend. {hint}"}
        except Exception as e:
            logger.error("Chatbot error: %s", e)
            return {"success": False, "error": f"Something went wrong: {e}"}


_chatbot_service_instance: Optional[AgriChatbotService] = None


def get_chatbot_service(**kwargs) -> AgriChatbotService:
    global _chatbot_service_instance
    if _chatbot_service_instance is None:
        _chatbot_service_instance = AgriChatbotService(**kwargs)
    return _chatbot_service_instance
