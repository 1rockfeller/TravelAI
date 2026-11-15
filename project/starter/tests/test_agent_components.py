import asyncio
import json
from types import SimpleNamespace

import pytest

from app import main
from app.eval import judge as evaluation
from app.eval.llm_judge import LLMJudge
from app.long_term_memory import core as memory_core
from app.memory import ShortTermMemory
from app.rag import ingest, retriever
from app.state import AgentState, Phase
from app.tools import knowledge


class FakeContainer:
    def __init__(self):
        self.items = {}
        self.query_arguments = None

    def create_item(self, body):
        self.items[body["id"]] = dict(body)
        return body

    def read_item(self, item, partition_key):
        stored = self.items[item]
        assert stored["session_id"] == partition_key
        return dict(stored)

    def upsert_item(self, body):
        self.items[body["id"]] = dict(body)
        return body

    def query_items(self, **kwargs):
        self.query_arguments = kwargs
        if kwargs.get("query") == "SELECT VALUE COUNT(1) FROM c":
            return iter([len(self.items)])
        return iter([{
            "id": "snippet-1",
            "text": "BankGold dining benefits",
            "pk": "cards",
            "similarity_score": 0.87,
        }])


def test_agent_state_advances_through_done_and_stops():
    state = AgentState()

    for phase in list(Phase)[1:]:
        assert state.advance() is True
        assert state.phase is phase

    assert state.is_complete()
    assert state.advance() is False


def test_agent_state_reset_clears_all_tracking():
    state = AgentState()
    old_session_id = state.session_id
    state.advance()
    state.destination = "Paris"
    state.tools_called.append("weather")
    state.add_issue("test issue")
    state.set_structured_output({"destination": "Paris"})

    state.reset()

    assert state.phase is Phase.Init
    assert state.session_id != old_session_id
    assert state.destination is None
    assert state.tools_called == []
    assert state.issues == []
    assert state.structured_output is None


def test_create_kernel_registers_services_and_plugins(monkeypatch):
    for name, value in {
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
        "AZURE_OPENAI_KEY": "test-key",
        "AZURE_OPENAI_API_VERSION": "2024-02-15-preview",
        "AZURE_OPENAI_CHAT_DEPLOYMENT": "chat",
        "AZURE_OPENAI_EMBED_DEPLOYMENT": "embed",
    }.items():
        monkeypatch.setenv(name, value)

    kernel = main.create_kernel()

    assert {"weather", "fx", "search", "cards", "knowledge"}.issubset(
        set(kernel.plugins.keys())
    )
    assert kernel.get_service(service_id="chat") is not None
    assert kernel.get_service(service_id="embeddings") is not None


def test_run_request_returns_validated_plan_and_keeps_history_once(monkeypatch):
    response_plan = {
        "destination": "Paris",
        "travel_dates": "2026-06-01 to 2026-06-08",
        "card_recommendation": None,
        "currency_info": None,
        "results": [{
            "title": "Le Comptoir",
            "snippet": "A Paris restaurant",
            "url": "https://example.test/restaurant",
        }],
        "next_steps": [],
    }

    class FakeChatService:
        captured_history = None

        async def get_chat_message_contents(self, chat_history, **kwargs):
            self.captured_history = chat_history
            return [SimpleNamespace(content=json.dumps(response_plan))]

    service = FakeChatService()

    class FakeKernel:
        def get_service(self, **kwargs):
            return service

    monkeypatch.setattr(main, "validate_all_config", lambda: {})
    monkeypatch.setattr(main, "create_kernel", lambda: FakeKernel())
    memory = ShortTermMemory()
    memory.add_conversation("user", "Earlier request")
    memory.add_conversation("assistant", "Earlier response")

    result = json.loads(asyncio.run(main.run_request(
        "Plan a Paris trip",
        memory=memory,
    )))

    assert result["trip_plan"]["destination"] == "Paris"
    assert result["trip_plan"]["citations"] == ["https://example.test/restaurant"]
    assert result["metadata"]["data_quality"] == "validated_with_pydantic"
    user_messages = [
        message.content
        for message in service.captured_history.messages
        if getattr(message.role, "value", message.role) == "user"
    ]
    assert user_messages.count("Plan a Paris trip") == 1
    assert "Earlier request" in user_messages


def test_llm_judge_parses_numeric_scores_without_inventing_fallback():
    judge = LLMJudge(main.Kernel())
    evaluation = judge._parse_evaluation_result(json.dumps({
        "criteria_scores": {
            "accuracy": 4,
            "completeness": 3,
            "relevance": 4,
            "tool_usage": 5,
            "structure": 4,
            "citations": 3,
        },
        "reasoning": "Supported by the supplied response.",
        "recommendations": [],
    }))

    assert abs(evaluation.overall_score - 3.85) < 1e-9
    assert evaluation.passed is True
    assert evaluation.criteria_scores["tool_usage"] == 5

    try:
        judge._parse_evaluation_result("A good response.")
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("Malformed judge output must not receive a made-up score")


def test_failed_agent_request_is_not_reported_as_judge_score(monkeypatch):
    async def failed_request(_):
        return json.dumps({"status": "failed", "error": "service unavailable"})

    monkeypatch.setattr(evaluation, "run_request", failed_request)
    with pytest.raises(RuntimeError, match="service unavailable"):
        asyncio.run(evaluation.evaluate({"name": "test", "query": "Paris"}, object()))


def test_llm_judge_propagates_service_errors():
    class BrokenKernel:
        services = {}

        def get_service(self, **kwargs):
            raise RuntimeError("judge service unavailable")

    judge = LLMJudge(BrokenKernel())
    with pytest.raises(RuntimeError, match="judge service unavailable"):
        asyncio.run(judge.evaluate_response(
            user_query="Paris", agent_response="response", structured_output={},
            tool_calls=[], citations=[],
        ))


def test_long_term_memory_add_and_get_update_access(monkeypatch):
    container = FakeContainer()
    monkeypatch.setattr(memory_core, "get_cosmos_client", lambda **kwargs: None)
    monkeypatch.setattr(memory_core, "get_container", lambda: container)
    monkeypatch.setattr(memory_core, "get_openai_kernel", lambda enabled: None)

    memory = memory_core.LongTermMemory(enable_ai_scoring=False)
    memory_id = asyncio.run(memory.add_memory(
        session_id="session-1",
        content="Remember Paris is preferred",
        importance_score=0.8,
        tags=["travel"],
        metadata={"source": "test"},
        context="current trip",
    ))

    stored = container.items[memory_id]
    assert stored["importance_score"] == 0.8
    assert stored["metadata"] == {
        "source": "test",
        "context": "current trip",
    }

    retrieved = memory.get_memory(memory_id, "session-1")
    assert retrieved is not None
    assert retrieved.access_count == 1
    assert container.items[memory_id]["access_count"] == 1


def test_vector_retriever_embeds_and_runs_partition_scoped_query(monkeypatch):
    container = FakeContainer()
    monkeypatch.setattr(
        retriever, "embed_texts",
        lambda texts: asyncio.sleep(0, result=[[0.1, 0.2, 0.3]]),
    )
    monkeypatch.setattr(
        retriever, "get_cosmos_client", lambda: (object(), container)
    )

    results = asyncio.run(retriever.retrieve("dining benefits", k=2, partition_key="cards"))

    assert results[0]["id"] == "snippet-1"
    assert "SELECT TOP 2" in container.query_arguments["query"]
    assert container.query_arguments["parameters"] == [
        {"name": "@queryVector", "value": [0.1, 0.2, 0.3]},
        {"name": "@pk", "value": "cards"},
    ]
    assert container.query_arguments["partition_key"] == "cards"
    assert container.query_arguments["enable_cross_partition_query"] is False


def test_knowledge_orders_cosine_similarity_highest_first(monkeypatch):
    async def fake_retrieve(*args, **kwargs):
        return [
            {"id": "low", "text": "Less relevant", "similarity_score": 0.3},
            {"id": "high", "text": "More relevant", "similarity_score": 0.9},
        ]

    monkeypatch.setattr(knowledge, "retrieve", fake_retrieve)
    monkeypatch.setattr(knowledge, "search_card_benefits", lambda *args: [])
    results = asyncio.run(knowledge.KnowledgeTools().search_knowledge("test"))
    assert [item["content"] for item in results] == ["More relevant", "Less relevant"]


def test_snippet_ingestion_stores_embedding(monkeypatch):
    container = FakeContainer()
    monkeypatch.setattr(
        ingest, "embed_texts",
        lambda texts: asyncio.sleep(0, result=[[0.4, 0.5]]),
    )
    monkeypatch.setattr(ingest, "get_container", lambda: container)

    document = asyncio.run(ingest.upsert_snippet("snippet-2", "A test snippet"))

    assert document["embedding"] == [0.4, 0.5]
    assert container.items["snippet-2"]["pk"] == "cards"
