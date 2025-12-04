
import os
import time
from textwrap import shorten
from typing import Any, Iterable, List

from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env (GROQ_API_KEY, GROQ_MODEL)



class RAGGenerator:
    """Improved retrieval-augmented generator for product recommendations.

    Key improvements:
    - remove duplicate results
    - rerank by simple metadata heuristics
    - early-stop for empty/short queries
    - retry logic for Groq API calls
    - stronger prompt instructions to reduce hallucinations
    """

    # Prompt used to guide the LLM. Clear instructions reduce hallucination.
    PROMPT_TEMPLATE = """
You are an expert e-commerce assistant. Use ONLY the metadata provided below.

## Your tasks
1. Compare the products using their metadata (price, discount, rating, stock, short description).
2. Recommend the single best product for the user, explaining trade-offs (price vs. quality, availability, rating, etc.).
3. When possible, suggest exactly one alternative that serves a different need (e.g., cheaper, premium, or better rated).
4. If none of the retrieved products fit the request, output exactly:
   There are no good products that fit the request based on the retrieved results.
5. Always reference products by PID and title.
6. Do NOT invent attributes that are not present in the metadata. If something is unknown, say "Unknown".

## Retrieved Products
{retrieved_results}

## User Request
{user_query}

## Respond with
- Best Product: PID – Title
- Why: cite concrete attributes (price, rating, discount, stock, description)
- Alternative (optional): PID – Title + justification
"""

    MAX_RESULT_DESC = 220  # max description length to keep the prompt compact
    MIN_QUERY_LEN = 2      # minimum query length to consider calling the LLM
    DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."

    def __init__(self) -> None:
        # Cache the Groq client to avoid repeated initialization
        self._client: Groq | None = None


    # Public API
    def generate_response(
        self,
        user_query: str,
        retrieved_results: Iterable[Any],
        top_N: int = 5
    ) -> str:
        """Generate a recommendation string for the user query using the provided hits.

        Performs simple input validation, deduplication and reranking before calling Groq.
        """

        user_query = (user_query or "").strip()

        # Early exit for too-short queries
        if len(user_query) < self.MIN_QUERY_LEN:
            return "There are no good products that fit the request based on the retrieved results."

        hits = list(retrieved_results or [])
        if not hits:
            return "There are no good products that fit the request based on the retrieved results."

        # Apply deduplication and a lightweight reranking heuristic, then limit
        hits = self._dedupe_by_pid(hits)
        hits = self._rerank_results(hits)[:top_N]

        if not hits:
            return "There are no good products that fit the request based on the retrieved results."

        formatted_results = self._format_results(hits)
        prompt = self.PROMPT_TEMPLATE.format(
            retrieved_results=formatted_results,
            user_query=user_query or "(empty query)",
        )

        try:
            client = self._get_client()
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            # Call the LLM with retry/backoff to handle transient errors
            chat_completion = self._call_with_retry(
                client,
                model_name=model_name,
                prompt=prompt,
                retries=3,
                backoff_sec=0.7,
            )

            generation = chat_completion.choices[0].message.content
            return generation.strip() if generation else self.DEFAULT_ANSWER

        except Exception as e:
            import traceback
            print("RAG ERROR:", e)
            traceback.print_exc()
            # Do not expose secrets in logs in production
            print("GROQ model:", os.environ.get("GROQ_MODEL"))
            return self.DEFAULT_ANSWER



    #groq client + retry handling

    def _get_client(self) -> Groq:
        """Return a cached Groq client, creating it on first use.

        Raises RuntimeError if `GROQ_API_KEY` is not set.
        """
        if self._client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set in environment.")
            self._client = Groq(api_key=api_key)
        return self._client

    def _call_with_retry(
        self,
        client: Groq,
        model_name: str,
        prompt: str,
        retries: int = 3,
        backoff_sec: float = 0.7,
    ):
        """Call Groq with simple retry/backoff for transient failures."""
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                return client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                )
            except Exception as e:
                last_err = e
                if attempt == retries - 1:
                    raise
                time.sleep(backoff_sec * (attempt + 1))

        raise last_err or RuntimeError("Unknown error during Groq call")


    #reranking and deduplication 
    def _dedupe_by_pid(self, results: List[Any]) -> List[Any]:
        """Remove duplicates based on `pid` to keep the prompt concise."""
        seen = set()
        unique: List[Any] = []
        for res in results:
            pid = self._get_attr(res, "pid")
            if pid and pid not in seen:
                seen.add(pid)
                unique.append(res)
        return unique or results

    def _rerank_results(self, results: List[Any]) -> List[Any]:
        """Simple heuristic reranking: prefer higher rating, in-stock and lower price."""

        def score(res: Any) -> float:
            rating = self._safe_float(self._get_attr(res, "average_rating"), default=3.0)
            price = self._safe_float(self._get_attr(res, "selling_price"), default=50.0)
            out_of_stock = bool(self._get_attr(res, "out_of_stock"))
            stock_bonus = 1.0 if not out_of_stock else 0.0

            # Formula semplice migliorabile: rating pesa forte, prezzo penalizza
            return rating * 2.0 + stock_bonus * 3.0 - price * 0.02

        return sorted(results, key=score, reverse=True)


    #context formatting for the prompt
    def _format_results(self, results: List[Any]) -> str:
        """Format products into a compact text block for the model."""
        rows = []
        for idx, res in enumerate(results, start=1):
            pid = self._get_attr(res, "pid") or "N/A"
            title = self._get_attr(res, "title") or "Unknown"
            price = self._format_currency(self._get_attr(res, "selling_price"))
            discount = self._get_attr(res, "discount")
            rating = self._get_attr(res, "average_rating")
            availability = "Out of stock" if self._get_attr(res, "out_of_stock") else "In stock"
            description = shorten((self._get_attr(res, "description") or "N/A").strip(), self.MAX_RESULT_DESC)

            row = (
                f"{idx}. PID {pid} — {title}\n"
                f"   • Price: {price} | Discount: {self._format_discount(discount)} | "
                f"Rating: {self._format_rating(rating)} | {availability}\n"
                f"   • Description: {description}"
            )
            rows.append(row)
        return "\n".join(rows)


    #generic utilities
    @staticmethod
    def _get_attr(obj: Any, key: str) -> Any:
        """Support both object attributes and dict keys."""
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key)
        return None

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Robustly convert a value to float, returning `default` on error."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _format_currency(value: Any) -> str:
        try:
            return f"${float(value):.2f}"
        except (TypeError, ValueError):
            return "N/A"

    @staticmethod
    def _format_discount(value: Any) -> str:
        try:
            val = float(value)
            if val <= 0:
                return "0%"
            return f"{val:.0f}%"
        except (TypeError, ValueError):
            return "N/A"

    @staticmethod
    def _format_rating(value: Any) -> str:
        try:
            return f"{float(value):.1f}/5"
        except (TypeError, ValueError):
            return "N/A"
