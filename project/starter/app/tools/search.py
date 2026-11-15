import json
import logging
import os
from typing import Any, Dict, List, Optional

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI
from semantic_kernel.functions import kernel_function

logger = logging.getLogger(__name__)


class SearchTools:
    def __init__(self):
        self.project_endpoint = os.getenv("PROJECT_ENDPOINT")
        self.agent_name = os.getenv("AGENT_ID")

    @kernel_function(
        name="web_search",
        description="Search the web for current restaurants, hotels, attractions, and travel information.",
    )
    def web_search(
        self,
        query: str,
        max_results: int = 5,
        filter_keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search through an Azure AI Agent configured with Bing grounding."""
        if not all([self.project_endpoint, self.agent_name]):
            missing = [
                name
                for name, value in (
                    ("PROJECT_ENDPOINT", self.project_endpoint),
                    ("AGENT_ID", self.agent_name),
                )
                if not value
            ]
            return [{
                "title": "Missing configuration",
                "url": "",
                "snippet": f"Set the required search settings: {', '.join(missing)}",
                "error": True,
            }]
        if not query.strip():
            raise ValueError("Search query must not be empty")
        if not 1 <= max_results <= 10:
            raise ValueError("max_results must be between 1 and 10")

        credential = DefaultAzureCredential()
        openai_client = None
        try:
            prompt = (
                "Use the configured Bing grounding tool to search for the following. "
                f"Return at most {max_results} useful results. "
                "Respond with only a JSON array, each item having string fields "
                "title, url, and snippet. Do not invent sources or URLs.\n\n"
                f"Search query: {query}"
            )
            openai_client = OpenAI(
                base_url=f"{self.project_endpoint.rstrip('/')}/openai/v1",
                api_key=get_bearer_token_provider(
                    credential, "https://ai.azure.com/.default"
                ),
            )
            response = openai_client.responses.create(
                input=prompt,
                extra_body={
                    "agent_reference": {
                        "name": self.agent_name,
                        "type": "agent_reference",
                    }
                },
            )
            if response.status != "completed":
                raise RuntimeError(
                    f"Azure AI search agent finished with status {response.status}"
                )
            raw_text = response.output_text
            if not raw_text:
                raise RuntimeError("Azure AI search agent returned no text content")

            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                if cleaned.rstrip().endswith("```"):
                    cleaned = cleaned.rstrip()[:-3].strip()
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start < 0 or end < start:
                raise ValueError("Search response did not contain a JSON array")
            results = json.loads(cleaned[start:end + 1])
            if not isinstance(results, list):
                raise ValueError("Search response JSON must be an array")

            normalized = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                normalized.append({
                    "title": str(result.get("title") or "Unknown"),
                    "url": str(result.get("url") or ""),
                    "snippet": str(result.get("snippet") or ""),
                })

            if filter_keywords:
                keywords = [word.lower() for word in filter_keywords]
                normalized = [
                    result for result in normalized
                    if any(
                        keyword in (result["title"] + " " + result["snippet"]).lower()
                        for keyword in keywords
                    )
                ]
            return normalized[:max_results]
        except Exception as exc:
            logger.exception("Bing-grounded search failed")
            return [{
                "title": "Search error",
                "url": "",
                "snippet": f"Failed to retrieve search results: {exc}",
                "error": True,
            }]
        finally:
            if openai_client is not None:
                openai_client.close()
            credential.close()
