import os
from textwrap import shorten
from typing import Any, Iterable, List

from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env


class RAGGenerator:
    """Retrieval-Augmented Generator tuned for product comparisons."""

    PROMPT_TEMPLATE = """
You are an expert e-commerce stylist. Analyze the retrieved products and craft a concise recommendation.

## What to do
1. Compare the products using THEIR metadata (price, discount, stock, rating, description snippet).
2. Recommend the best product for the user, explaining the trade-offs (price vs. quality, availability, brand fit, etc.).
3. When possible, suggest exactly one alternative that serves a different need (e.g., cheaper, premium, or better rated).
4. If none of the retrieved products fit the request, output exactly: There are no good products that fit the request based on the retrieved results.
5. Always reference products by PID and title.

## Retrieved Products
{retrieved_results}

## User Request
{user_query}

## Respond with
- Best Product: PID – Title
- Why: cite concrete attributes
- Alternative (optional): PID – Title + justification
"""

    MAX_RESULT_DESC = 220

    def __init__(self) -> None:
        self._client: Groq | None = None

    def generate_response(self, user_query: str, retrieved_results: Iterable[Any], top_N: int = 5) -> str:
        """Generate the RAG summary for the given query and search hits."""

        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."

        hits = list(retrieved_results or [])
        if not hits:
            return "There are no good products that fit the request based on the retrieved results."

        formatted_results = self._format_results(hits[:top_N])

        try:
            client = self._client or Groq(api_key=os.environ.get("GROQ_API_KEY"))
            self._client = client
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            prompt = self.PROMPT_TEMPLATE.format(
                retrieved_results=formatted_results,
                user_query=user_query.strip() or "(empty query)",
            )

            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
            )

            generation = chat_completion.choices[0].message.content
            return generation.strip() if generation else DEFAULT_ANSWER
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return DEFAULT_ANSWER

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_results(self, results: List[Any]) -> str:
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
                f"   • Price: {price} | Discount: {self._format_discount(discount)} | Rating: {self._format_rating(rating)} | {availability}\n"
                f"   • Description: {description}"
            )
            rows.append(row)
        return "\n".join(rows)

    @staticmethod
    def _get_attr(obj: Any, key: str) -> Any:
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key)
        return None

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
