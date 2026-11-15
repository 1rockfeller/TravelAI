# app/main.py - Travel Concierge Agent with Semantic Kernel
import os
import json
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding, OpenAIChatPromptExecutionSettings
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory
from app.state import AgentState
from app.memory import ShortTermMemory
from app.utils.config import validate_all_config
from app.utils.logger import setup_logger
from app.tools.weather import WeatherTools
from app.tools.fx import FxTools
from app.tools.search import SearchTools
from app.tools.card import CardTools
from app.tools.knowledge import KnowledgeTools
from app.models import TripPlan
import json as json_module

logger = setup_logger("travel_agent")





# ------------------------------
# KERNEL CREATION
# ------------------------------
def create_kernel():
    """
    Create and configure the Semantic Kernel instance.

    Create the Azure OpenAI services and register all concierge plugins.
    """
    kernel = Kernel()
    service_config = {
        "endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_key": os.environ["AZURE_OPENAI_KEY"],
        "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
    }
    kernel.add_service(
        AzureChatCompletion(
            deployment_name=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            service_id="chat",
            **service_config,
        )
    )
    kernel.add_service(
        AzureTextEmbedding(
            deployment_name=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"],
            service_id="embeddings",
            **service_config,
        )
    )
    for plugin_name, plugin in (
        ("weather", WeatherTools()),
        ("fx", FxTools()),
        ("search", SearchTools()),
        ("cards", CardTools()),
        ("knowledge", KnowledgeTools()),
    ):
        kernel.add_plugin(plugin, plugin_name=plugin_name)

    return kernel


# -------------------------------
# MAIN REQUEST PIPELINE (AGENTIC LOOP)
# -------------------------------
async def run_request(user_input: str, memory: ShortTermMemory = None) -> str:
    try:
        validate_all_config()
        kernel = create_kernel()

        # Initialize state machine
        state = AgentState()
        if not user_input.strip():
            raise ValueError("User input must not be empty")

        def advance_state():
            if not state.advance():
                raise RuntimeError("Cannot advance a completed agent state")
            logger.info(f"📍 State: {state.phase.value}")

        logger.info(f"📍 State: {state.phase}")

        # Initialize or use existing short-term memory
        if memory is None:
            memory = ShortTermMemory(max_items=10, max_tokens=4000)
        state.session_id = memory.session_id
        previous_history = memory.get_conversation_history()
        memory.add_conversation("user", user_input)

        advance_state()
        state.set_requirements({"user_input": user_input})
        logger.info(f"Request: {user_input}")

        # Get chat service
        chat_service = kernel.get_service(type=ChatCompletionClientBase)

        # Create chat history
        chat_history = ChatHistory()

        # ------------------------------
        # SYSTEM MESSAGE PROMPT
        # ------------------------------
        system_message = """You are a professional AI travel concierge. Help users plan
trips using the registered tools and return only one valid JSON object matching the
TripPlan schema below; do not wrap it in Markdown or add text outside the JSON.

Available tools:
- weather.get_weather(city): obtain a current seven-day forecast for a city.
- search.web_search(query, max_results): find current restaurants, hotels,
  attractions, and travel information. Use returned titles, snippets, and URLs only.
- fx.convert_fx(amount, base, target): convert currency using a current exchange rate.
- cards.recommend_card(category, country, amount): compare the supported cards.
- knowledge.search_knowledge(query, card_name, category, country): find card benefits
  and policies in the knowledge base. Use the user's card name when provided.
- knowledge.get_card_recommendation(card_name, category, amount, country): calculate
  benefits for a card the user already holds.
- knowledge.get_lounge_access(airport_code, card_name): check lounge information.
- knowledge.get_travel_tips(country, category): retrieve country-specific guidance.

For a trip request, use weather for the destination; web search for requested
restaurants, lodging, or attractions; and FX conversion when a local-currency
estimate is useful. For card questions, search the knowledge base and use the exact
card benefit data. You may call multiple tools. Do not claim a tool succeeded if its
result reports an error. Ask for clarification through next_steps if essential
information is missing.

Anti-hallucination rules:
- Only include facts returned by tools or explicitly supplied by the user.
- Never invent ratings, prices, weather, exchange rates, card terms, citations, or URLs.
- Use null for unavailable optional objects and lists; use "N/A" for an unknown
  destination or travel_dates. Use [] for next_steps when none are needed.
- Always include destination, travel_dates, card_recommendation and currency_info
  in the JSON, even when their values are unknown or null.
- Include citations only for source URLs actually returned by web search.
- Keep search result fields limited to the information that was actually returned.

TripPlan JSON schema:
{
  "destination": "string",
  "travel_dates": "string",
  "weather": {
    "temperature_c": "number or null",
    "conditions": "string or null",
    "recommendation": "string or null"
  } or null,
  "results": [
    {
      "title": "string",
      "snippet": "string or null",
      "url": "string or null",
      "price_range": "string or null",
      "rating": "number or null",
      "category": "string or null"
    }
  ] or null,
  "card_recommendation": {
    "card": "string",
    "benefit": "string",
    "fx_fee": "string",
    "source": "string"
  } or null,
  "currency_info": {
    "usd_to_eur": "number or null",
    "sample_meal_usd": "number or null",
    "sample_meal_eur": "number or null",
    "points_earned": "integer or null"
  } or null,
  "citations": ["string"] or null,
  "next_steps": ["string"]
}
"""

        chat_history.add_system_message(system_message)

        # Add prior turns only; the current request is already appended below.
        for item in previous_history:
            if item.get("role") == "user":
                chat_history.add_user_message(item.get("content", ""))
            elif item.get("role") == "assistant":
                chat_history.add_assistant_message(item.get("content", ""))

        chat_history.add_user_message(user_input)

        # Enable automatic function calling
        execution_settings = OpenAIChatPromptExecutionSettings(
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        advance_state()
        advance_state()
        logger.info("🤖 LLM will automatically call tools as needed...")

        # Let LLM automatically call tools
        response = await chat_service.get_chat_message_contents(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel
        )

        advance_state()

        agent_response = response[0].content or ""
        logger.info(f"✅ Agent response received: {len(agent_response)} chars")

        # Parse and validate response with Pydantic (Lesson 2 pattern)
        try:
            # Extract JSON from response (handle cases where LLM includes extra text)
            json_start = agent_response.find('{')
            json_end = agent_response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = agent_response[json_start:json_end]
            response_data = json_module.loads(json_str)

            logger.info("✅ JSON parsed successfully")

            # Validate with TripPlan Pydantic model
            trip_plan = TripPlan.model_validate(response_data)
            logger.info(f"✅ Pydantic validation passed: {trip_plan.destination}")

            # Auto-populate citations from search results if empty
            if (not trip_plan.citations or trip_plan.citations == []) and trip_plan.results:
                trip_plan.citations = [r.url for r in trip_plan.results if r.url]
                logger.info(f"✅ Auto-populated {len(trip_plan.citations)} citations from results")

            state.set_structured_output(trip_plan.model_dump())
            for citation in trip_plan.citations or []:
                state.add_citation(citation)
            advance_state()
            advance_state()
            advance_state()

            memory.add_conversation("assistant", agent_response)
            tool_calls = []
            for message in getattr(chat_history, "messages", []):
                for content_item in getattr(message, "items", []):
                    function_name = getattr(content_item, "function_name", None)
                    if function_name:
                        tool_calls.append(function_name)
            state.tools_called = list(dict.fromkeys(tool_calls))

            # Return validated Pydantic model as JSON
            result = {
                "trip_plan": trip_plan.model_dump(),
                "metadata": {
                    "session_id": state.session_id,
                    "tools_called": state.tools_called,
                    "data_quality": "validated_with_pydantic",
                    "memory_items": len(memory.get_conversation_history())
                }
            }

            return json.dumps(result, indent=2, default=str)

        except Exception as e:
            logger.exception("Agent response could not be validated as TripPlan")
            memory.add_conversation("assistant", agent_response)
            state.add_issue(str(e))
            advance_state()
            advance_state()
            advance_state()
            return json.dumps({
                "error": f"Agent response failed TripPlan validation: {e}",
                "status": "failed",
                "metadata": {
                    "session_id": state.session_id,
                    "phase": state.phase.value,
                    "data_quality": "validation_failed",
                },
            }, indent=2, default=str)

    except Exception as e:
        logger.exception("Request failed")
        return json.dumps({"error": str(e), "status": "failed"}, indent=2)



# -------------------------------
# CLI ENTRY POINT
# -------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Travel Concierge Agent")
    parser.add_argument("--input", help="User input for the agent")
    args = parser.parse_args()

    if args.input:
        result = asyncio.run(run_request(args.input))
        print(result)
    else:
        print("Travel Concierge Agent (type 'quit' to exit)")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                result = asyncio.run(run_request(user_input))
                try:
                    data = json.loads(result)
                    if "plan" in data:
                        from app.utils.pretty_print import print_plan  # 👈 optional: factor out
                        print_plan(data["plan"])
                    else:
                        print(result)
                except Exception:
                    print(result)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()