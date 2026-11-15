# Travel Concierge Agent

> **📖 Purpose**: This is the main project README providing a high-level overview and quick start guide for the Travel Concierge Agent. Use this file to get started quickly and understand the project at a glance.

An intelligent travel planning agent built with Microsoft Semantic Kernel, Azure OpenAI, and Cosmos DB for vector RAG.

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or later
- Azure OpenAI service with gpt-4.1-mini and text-embedding-3-small deployments
- Cosmos DB account with vector search enabled
- A Foundry agent configured with Grounding with Bing Search and project access via `az login`

### Installation
```bash
cd project/starter
pip install -r requirements.txt
cp env.example .env
# Edit .env with your Azure credentials
```

### Usage

#### 💬 Chat Interface (Recommended)
```bash
python chat.py
```

#### Programmatic Usage
```python
from app.main import run_request
import json
import asyncio

result = asyncio.run(run_request("I want to go to Paris from 2026-06-01 to 2026-06-08 with my BankGold card"))
plan_data = json.loads(result)
print(plan_data["trip_plan"]["destination"])
```

## 🏗️ Architecture

- **Semantic Kernel**: Tool orchestration, state management, and service integration (AzureChatCompletion and AzureTextEmbedding)
- **Azure OpenAI**: gpt-4.1-mini for chat and text-embedding-3-small for embeddings, accessed via Semantic Kernel services
- **Memory Systems**: Short-term and long-term memory for context management
- **Cosmos DB**: Vector RAG for long-term memory and knowledge retrieval
- **Class-Based Tools**: Modular tool architecture (WeatherTools, FxTools, SearchTools, CardTools)

The Cosmos DB account must have NoSQL vector search enabled before creating the
snippets container. The ingestion script configures a 1,536-dimension cosine
embedding policy by default; set `AZURE_OPENAI_EMBED_DIMENSIONS` if the deployed
embedding model uses another dimension. An existing container without the policy
can return brute-force similarity results, but it does not have the vector index
created by this script. Vector policies cannot be added to an existing container:
create a separate vector-enabled container and migrate the data if indexing is
required. Check cost and do not delete the original container without a backup.

### Submission evidence

Run these commands from `project/starter` (Azure calls consume resources):

```powershell
python -X utf8 -m app.scripts.system_check
python -X utf8 -m app.rag.retriever "BankGold dining benefits"
python -X utf8 -m app.eval.judge
python -m pytest tests/test_memory.py tests/test_models.py tests/test_state.py -v
```

The retrieval command displays embedding dimensions, the `VectorDistance`
results, cosine similarity scores, and the retrieved text. The judge command
also displays the agent's state transitions and writes `app/eval/results.csv`.
An evaluation error is reported as an error, not a numerical score.

The web-search client calls the current named Foundry agent through the
Responses API. Its Bing connection is configured on the agent, not passed as
`BING_CONNECTION_ID` to `AIProjectClient`. This differs from the rubric's
literal classic-SDK requirement; verify acceptance with the reviewer and verify
the tool and connection in the Azure portal. A separate Azure AI Search
connection, if required by the rubric, is not provisioned by this project.

## 🛠️ Development

### Testing & Progress Checking

#### **System Health Check**
```bash
# Comprehensive system health check
python -m app.scripts.system_check
```
This will verify:
- ✅ Azure OpenAI connectivity
- ✅ Cosmos DB connection
- ✅ All tool functionality
- ✅ Environment configuration
- ✅ Memory systems

#### **Unit Testing**
```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_models.py -v      # Data models
python -m pytest tests/test_tools.py -v       # Tool functionality
python -m pytest tests/test_state.py -v       # State management
python -m pytest tests/test_memory.py -v      # Memory systems
```

#### **Comprehensive Test Suite**
```bash
# Run all tests including health checks
python run_tests.py

# Run with different options
python run_tests.py --unit --verbose          # Unit tests only
python run_tests.py --quick                   # Quick tests only
```

#### **Progress Tracking**
- **Green checkmarks (✅)** in system_check.py output indicate working components
- **All tests passing** means your system is ready for development
- **Failed tests** will show specific issues to fix

### Project Structure
```
app/
├── main.py                 # Main entry point with SK integration
├── models.py              # Pydantic schemas
├── knowledge_base.py      # Card knowledge base data
├── state.py               # Enhanced agent state machine (8 phases)
├── memory.py              # Short-term memory system
├── long_term_memory/      # Long-term memory with Cosmos DB
├── tools/                 # Class-based tool implementations
├── rag/                   # Vector RAG system
├── eval/                  # Evaluation harness
└── utils/                 # Utility modules
```

## 📚 Documentation

- **[app/README.md](app/README.md)** - Complete technical reference with API docs, data models, and tool functions
- **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)** - Step-by-step learning guide for students with development phases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section in app/README.md
- Review the evaluation results
- Run health checks
- Check Azure service status
