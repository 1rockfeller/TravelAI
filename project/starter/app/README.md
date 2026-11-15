# Travel Concierge Agent - Technical Documentation

> **📖 Purpose**: This is the comprehensive technical documentation for the Travel Concierge Agent. Use this file for detailed API reference, data models, tool functions, configuration, and troubleshooting. This is your complete reference guide for development.

An intelligent travel planning agent built with Microsoft Semantic Kernel, Azure OpenAI, and Cosmos DB for vector RAG.

## 🏗️ Architecture

### Core Components
- **Semantic Kernel**: Tool orchestration, state management, and service integration (AzureChatCompletion and AzureTextEmbedding)
- **Azure OpenAI**: gpt-4.1-mini for chat and text-embedding-3-small for embeddings, accessed via Semantic Kernel services
- **Memory Systems**: Short-term and long-term memory for context management
- **Cosmos DB**: Vector RAG for long-term memory and knowledge retrieval
- **External APIs**: Open-Meteo (weather), Frankfurter (FX), Bing (search)
- **Class-Based Tools**: Modular tool architecture (WeatherTools, FxTools, SearchTools, CardTools)

### Agent State Machine
```
Init → ClarifyRequirements → PlanTools → ExecuteTools → AnalyzeResults
     → ResolveIssues → ProduceStructuredOutput → Done
```

The workflow initializes a session, clarifies requirements, plans and executes
tool calls, analyzes the results, resolves response issues, validates structured
output, and then completes the request.

## 🚀 Quick Start

### Prerequisites
1. **Azure Resources**:
   - Azure OpenAI service with GPT-4.1-mini and text-embedding-3-small deployments
   - Cosmos DB account with vector search enabled
   - Azure Key Vault (optional, for secrets management)
2. **Python**: 3.10 or later (required by Semantic Kernel 1.x)

2. **Environment Variables** (create `.env` file):
```bash
# Required Environment Variables
PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
AGENT_ID=your_foundry_agent_name
MODEL_DEPLOYMENT_NAME=your_model_deployment_name
AZURE_OPENAI_ENDPOINT=https://your-aoai.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_KEY=your_azure_openai_key_here
COSMOS_ENDPOINT=https://your-cosmos.documents.azure.com:443/
COSMOS_KEY=your_cosmos_key_here
COSMOS_DB=ragdb
COSMOS_CONTAINER=snippets
COSMOS_PARTITION_KEY=/pk
PYTHONPATH=.
```

`AGENT_ID` is the name of an existing Foundry agent configured with Grounding
with Bing Search, not a classic `asst_...` ID. Sign in with `az login` using an
identity with access to the project. The agent's tool configuration supplies
the Bing connection; `BING_CONNECTION_ID` is not used by the search client.

### Installation
```bash
pip install -r requirements.txt
```

### Usage

#### 💬 Chat Interface (Recommended)
The easiest way to interact with the agent:

```bash
# From project/solution directory
python chat.py
```

**Features:**
- Natural conversation interface
- Formatted output display
- Interactive commands (`help`, `status`, `clear`, `quit`)
- Error handling and recovery

#### Programmatic Usage

#### Basic Usage
```python
import asyncio
from app.main import run_request
import json

# Plan a trip using natural language
result = asyncio.run(run_request("I want to go to Paris from 2026-06-01 to 2026-06-08 with my BankGold card"))
plan_data = json.loads(result)
print(plan_data["trip_plan"]["destination"])  # "Paris"
```

#### Command Line
```bash
# Run the agent
python -m app.main

# Run evaluation
python -X utf8 -m app.eval.judge

# Demonstrate embedding generation and vector retrieval
python -m app.rag.retriever "BankGold dining benefits"

# Check system health
python -m app.scripts.system_check
```

## 🛠️ Tools & Plugins

### Weather Tool (`tools/weather.py`)
- **Source**: Open-Meteo API (free, no key required)
- **Function**: Get 7-day weather forecast
- **Class**: `WeatherTools` with `get_weather(lat, lon)` method

### FX Tool (`tools/fx.py`)
- **Source**: Frankfurter API (free, no key required)
- **Function**: Currency conversion
- **Class**: `FxTools` with `convert_fx(amount, base, target)` method

### Search Tool (`tools/search.py`)
- **Source**: Foundry agent configured with Grounding with Bing Search
- **Function**: Web search for restaurants and local info
- **Class**: `SearchTools` with `travel_web_search(query, max_results)` method

### Card Tool (`tools/card.py`)
- **Source**: In-memory rules engine
- **Function**: Credit card recommendations
- **Class**: `CardTools` with `recommend_card(mcc, amount, country)` method

### Knowledge Tool (`tools/knowledge.py`)
- **Source**: Cosmos DB vector search
- **Function**: Knowledge retrieval from curated snippets
- **Class**: `KnowledgeTools` with `get_card_recommendation(mcc, country)` method

### Memory Systems

#### **Short-Term Memory (`memory.py`)**
- **Purpose**: Session-based context management
- **Features**: Conversation history, tool calls, sliding window eviction
- **Storage**: In-memory with configurable limits
- **Use Cases**: Current conversation context, immediate recall

#### **Long-Term Memory (`long_term_memory.py`)**
- **Purpose**: Cross-session persistent memory
- **Features**: Importance scoring, pruning strategies, memory reordering
- **Storage**: Cosmos DB with vector search capabilities
- **Use Cases**: User preferences, historical context, learning

### RAG System (`rag/`)
- **Source**: Cosmos DB with vector search
- **Function**: Knowledge retrieval from curated snippets
- **Features**: Semantic search, embedding-based similarity
- **Components**: `ingest.py` for data ingestion, `retriever.py` for vector search

## 📊 Evaluation

The system includes a comprehensive evaluation harness:

```bash
python -m app.eval.judge
```

**Test Cases**:
1. **Paris Trip**: European destination with EUR currency
2. **Tokyo Trip**: Asian destination with JPY currency  
3. **Barcelona Trip**: Mediterranean destination with EUR currency

**Metrics**:
- ✅ Valid JSON output
- ✅ Citation presence
- ✅ Card recommendation accuracy
- ✅ Weather data quality
- ✅ Currency conversion accuracy

## 🔧 Development

### Project Structure
```
app/
├── main.py                 # Main entry point with SK integration
├── models.py              # Pydantic schemas (TripPlan, Weather, etc.)
├── synthesis.py           # AI synthesis and JSON generation
├── state.py               # Enhanced agent state machine (8 phases)
├── memory.py              # Short-term memory system
├── long_term_memory.py    # Long-term memory with Cosmos DB
├── knowledge_base.py      # Knowledge base for card recommendations
├── tools/                 # Class-based tool implementations
│   ├── weather.py         # WeatherTools class
│   ├── fx.py              # FxTools class
│   ├── search.py          # SearchTools class
│   ├── card.py            # CardTools class
│   └── knowledge.py       # KnowledgeTools class
├── rag/                   # Vector RAG system
│   ├── ingest.py          # Data ingestion
│   └── retriever.py       # Vector search
├── eval/                  # Evaluation harness
│   └── judge.py           # Simple rule-based evaluation
├── scripts/               # Utility scripts
│   └── system_check.py    # Comprehensive system health check
└── utils/                 # Utility modules
    ├── config.py          # Configuration management
    └── logger.py          # Logging setup
```

### Adding New Tools

1. **Create the tool class**:
```python
# tools/my_tool.py
from semantic_kernel.functions import kernel_function

class MyTool:
    @kernel_function(name="my_function", description="Does something useful")
    def my_function(self, param1: str, param2: int):
        # Implementation
        return {"result": "success"}
```

2. **Register in main.py**:
```python
from app.tools.my_tool import MyTool
kernel.add_plugin(MyTool(), "MyTool")
```

3. **Use in agent**:
```python
my_tool = kernel.get_plugin("MyTool")
result = my_tool["my_function"].invoke(kernel, param1="value", param2=42)
```

### State Management

The agent uses a state machine for robust execution:

```python
from app.state import AgentState, Phase

state = AgentState()
state.advance()  # Move to next phase
print(f"Current phase: {state.phase}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Cosmos DB Connection Error**:
   - Check `COSMOS_ENDPOINT` and `COSMOS_KEY` in `.env`
   - Verify RBAC permissions or use connection key
   - Ensure vector indexing is enabled

2. **Azure OpenAI Errors**:
   - Verify endpoint and deployment names
   - Check API key permissions
   - Ensure deployments exist and are active

3. **Tool Execution Errors**:
   - Check external API availability
   - Verify API keys (Bing, etc.)
   - Review tool function signatures

### Health Checks & Progress Monitoring

#### **System Health Check**
```bash
# Comprehensive system check
python -m app.scripts.system_check
```
This will verify:
- ✅ Azure OpenAI connectivity and deployments
- ✅ Cosmos DB connection and vector search
- ✅ All tool functionality (Weather, FX, Search, Card, Knowledge)
- ✅ Environment configuration validation
- ✅ Memory systems (short-term and long-term)
- ✅ RAG system functionality

#### **Unit Testing**
```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_models.py -v      # Data models validation
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

#### **Progress Tracking Guidelines**
- **System Check**: Look for green checkmarks (✅) - all should be green for a working system
- **Unit Tests**: All tests should pass - any failures indicate specific issues to fix
- **Test Coverage**: Run tests after each change to ensure nothing breaks
- **Debugging**: Failed tests will show specific error messages to help identify issues

## 📖 API Reference

### Core Functions

#### `run_request(user_input)`

Main entry point for generating travel plans using natural language input.

**Parameters:**
- `user_input` (str): Natural language description of travel requirements

**Returns:**
- `str`: JSON string containing the complete travel plan

**Example:**
```python
import asyncio
from app.main import run_request
import json

# Plan a trip using natural language
result = asyncio.run(run_request("I want to go to Tokyo from 2026-07-10 to 2026-07-17 with my BankGold card"))
plan_data = json.loads(result)
print(plan_data["trip_plan"]["destination"])  # "Tokyo"
```

### Data Models

#### `TripPlan`
Main data model for travel plans.

**Fields:**
- `destination` (str): Destination city
- `travel_dates` (str): Travel date range
- `weather` (Weather, optional): Weather information
- `results` (List[SearchResult], optional): Search results, including restaurants
- `card_recommendation` (CardRecommendation or null): Required field for the card recommendation
- `currency_info` (CurrencyInfo or null): Required field for currency conversion information
- `citations` (List[str], optional): Source citations
- `next_steps` (List[str]): Recommended next steps

#### `Weather`
Weather forecast information.

**Fields:**
- `temperature_c` (float): Temperature in Celsius
- `conditions` (str): Weather conditions description
- `recommendation` (str): Weather-based recommendations

#### `SearchResult`
Search result.

**Fields:**
- `title` (str): Result title
- `snippet` (str, optional): Result description
- `url` (str, optional): Result URL
- `price_range` (str, optional): Price range indicator
- `rating` (float, optional): Rating out of 5
- `category` (str, optional): Result category

#### `CardRecommendation`
Credit card recommendation.

**Fields:**
- `card` (str): Recommended card name
- `benefit` (str): Main benefit description
- `fx_fee` (str): Foreign exchange fee information
- `source` (str): Source of the recommendation

#### `CurrencyInfo`
Currency conversion information.

**Fields:**
- `usd_to_eur` (float, optional): USD to EUR exchange rate
- `sample_meal_usd` (float, optional): Sample meal cost in USD
- `sample_meal_eur` (float, optional): Sample meal cost in EUR
- `points_earned` (int, optional): Points earned for the transaction

### Tool Functions

#### Weather Tool
**Class**: `WeatherTools` with `get_weather(lat, lon)` method

Get weather forecast for coordinates.

**Example:**
```python
from app.tools.weather import WeatherTools

weather_tool = WeatherTools()
weather = weather_tool.get_weather(48.8566, 2.3522)  # Paris
print(weather['timezone'])  # "GMT"
```

#### FX Tool
**Class**: `FxTools` with `convert_fx(amount, base, target)` method

Convert currency using Frankfurter API.

**Example:**
```python
from app.tools.fx import FxTools

fx_tool = FxTools()
fx = fx_tool.convert_fx(100, "USD", "EUR")
print(fx['rates']['EUR'])  # 85.81
```

#### Search Tool
**Class**: `SearchTools` with `travel_web_search(query, max_results)` method

Search the web using Bing API.

**Example:**
```python
from app.tools.search import SearchTools

search_tool = SearchTools()
results = search_tool.travel_web_search("best restaurants Paris", 5)
print(len(results))  # 5
```

#### Card Tool
**Class**: `CardTools` with `recommend_card(mcc, amount, country)` method

Recommend credit card based on merchant category and location.

**Example:**
```python
from app.tools.card import CardTools

card_tool = CardTools()
card = card_tool.recommend_card("5812", 100, "France")
print(card['best']['card'])  # "BankGold"
```

#### Knowledge Tool
**Class**: `KnowledgeTools` with `get_card_recommendation(mcc, country)` method

Retrieve knowledge from curated snippets using vector search.

**Example:**
```python
from app.tools.knowledge import KnowledgeTools

knowledge_tool = KnowledgeTools()
knowledge = knowledge_tool.get_card_recommendation("5812", "France")
print(knowledge['card'])  # "BankGold"
```

### State Management

#### `AgentState`
Manages agent execution state.

**Properties:**
- `phase` (Phase): Current execution phase
- `destination` (str): Current destination
- `dates` (str): Current travel dates
- `card` (str): Current card type
- `tools_called` (List[str]): List of called tools

**Methods:**
- `advance()`: Move to next phase
- `reset()`: Reset to initial state

#### `Phase` Enum
Available execution phases:
- `Init`: Initial state
- `ClarifyRequirements`: Clarifying user requirements
- `PlanTools`: Planning tool execution
- `ExecuteTools`: Executing tools
- `Synthesize`: Synthesizing results
- `Done`: Final state

## 📈 Performance

- **Response Time**: ~2-5 seconds per request
- **Token Usage**: ~500-1000 tokens per synthesis
- **Memory**: Efficient with lazy loading and caching
- **Scalability**: Stateless design, ready for containerization

## 🔒 Security

- **Authentication**: Azure DefaultAzureCredential
- **Secrets**: Environment variables (use Key Vault in production)
- **Data**: No PII stored, only travel planning data
- **APIs**: All external APIs use HTTPS

## 📝 License

See [LICENSE.md](../../LICENSE.md) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the evaluation results
- Run health checks
- Check Azure service status