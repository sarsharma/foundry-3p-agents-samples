# Azure Foundry GPT-5-mini + LangGraph Agent

Forked from [this]( https://github.com/ninghu/foundry-3p-agents-samples/tree/main/aws/agent_core ) example

Currency-focused LangGraph workflow using Azure Foundry's GPT-5-mini model. The agent uses a `get_exchange_rate` tool to answer questions about currency exchange rates and optionally forwards telemetry to Azure Application Insights.

## Architecture

- **LLM**: Azure OpenAI GPT-5-mini (via Azure Foundry)
- **Framework**: LangGraph for agent orchestration
- **Web Server**: Flask + Gunicorn
- **Deployment**: Azure Container Apps ready
- **Monitoring**: Azure Application Insights (optional)

## Prerequisites

1. **Azure Foundry/Azure OpenAI** with GPT-5-mini deployed
2. **API credentials** from Azure Portal
3. **Python 3.12+** or Docker

## Configuration

All settings are loaded from environment variables. Copy `.env.example` to `.env` and fill in your values:

```bash
# Required
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-mini
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Optional - for tracing
APPLICATION_INSIGHTS_CONNECTION_STRING=InstrumentationKey=...
```

## Local Development

### Option 1: Python (Local)
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (or use .env file)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-5-mini"

# Run the agent
python langgraph_agent.py
```

The server will start on `http://localhost:8080`

### Option 2: Docker (Local)
```bash
# Build the image
docker build -t azure-foundry-agent .

# Run with environment variables
docker run -p 8080:8080 \
  -e AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
  -e AZURE_OPENAI_API_KEY="your-key" \
  -e AZURE_OPENAI_DEPLOYMENT_NAME="gpt-5-mini" \
  azure-foundry-agent
```

## API Usage


### Invoke Agent
```bash
curl -X POST http://localhost:8080/invoke \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the exchange rate from USD to EUR?"}'
```

Response:
```json
{
  "result": "Based on the latest data, 1 USD equals 0.92 EUR..."
}
```
