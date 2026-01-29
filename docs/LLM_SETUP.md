# LLM Configuration - Groq Setup

## Overview

The Hyperagentic Processor uses **Groq** as the LLM provider, exactly like your original class example from `groq_feed_agentic_tool_calling.py`. This ensures consistency with the patterns you learned in class.

## Model Configuration

**Primary Model**: `groq/llama-3.1-70b-versatile`
- Same model family as used in class examples
- 70B parameter model for high-quality reasoning
- Versatile variant optimized for diverse tasks
- Fast inference through Groq's infrastructure

## API Key Setup

### Option 1: Environment Variable (Recommended)
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### Option 2: .env File
Create or update `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Option 3: Direct Configuration
Edit `config/llm_config.py` to hardcode the key (not recommended for production).

## Configuration Details

The LLM configuration is centralized in `config/llm_config.py`:

```python
{
    "model": "groq/llama-3.1-70b-versatile",
    "api_key": os.getenv("GROQ_API_KEY"),
    "temperature": 0.7,
    "max_tokens": 4000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

## Usage in Agents

All motivated agents use this same configuration:

### ToolCreator Agent
- Uses Groq for natural language to code generation
- Temperature 0.7 for creative but focused code generation
- Max tokens 4000 for complete function implementations

### SafetyAgent
- Uses Groq for security analysis reasoning
- Combines LLM analysis with AST parsing
- Generates human-readable security explanations

### GradingAgent
- Uses Groq for performance evaluation reasoning
- Multi-dimensional analysis with detailed explanations
- Constructive feedback generation

## Connection to Class Examples

This setup mirrors your original class example:

**Original Class Code** (`groq_feed_agentic_tool_calling.py`):
```python
from groq import Groq
client = Groq()
completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",  # Different model
    messages=[...],
    tools=[...]
)
```

**Hyperagentic Processor**:
```python
from groq import Groq  # Same import
client = Groq()
completion = client.chat.completions.create(
    model="groq/llama-3.1-70b-versatile",  # Updated model
    messages=[...],
    tools=[...]
)
```

## Model Comparison

| Aspect | Class Example | Hyperagentic Processor |
|--------|---------------|------------------------|
| Provider | Groq | Groq ✅ |
| Model | openai/gpt-oss-120b | groq/llama-3.1-70b-versatile |
| Tool Calling | ✅ | ✅ |
| API Pattern | ✅ | ✅ |
| Retry Logic | ✅ | ✅ |

## Testing Without API Key

The system includes fallback configuration for testing:

```bash
# Test core functionality without API calls
python test_core_functionality.py

# Test with mock LLM responses
python test_functional_agents.py
```

## Validation

Check your setup:

```bash
cd hyperagentic_processor
python config/llm_config.py
```

Expected output:
```
✅ Groq LLM configuration validated
   Model: groq/llama-3.1-70b-versatile
   API Key: gsk_DhyEmiCRWGM5QPmD...
```

## Integration with AutoGen

The motivated agents integrate Groq with Microsoft AutoGen:

```python
class MotivatedAgent(AssistantAgent):
    def __init__(self, name, system_message, llm_config):
        # AutoGen agent with Groq backend
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config  # Groq configuration
        )
```

This gives you:
- AutoGen's multi-agent conversation management
- Groq's fast, high-quality inference
- Psychological drive integration
- Tool calling capabilities

## Why Groq?

1. **Consistency**: Same as your class examples
2. **Performance**: Fast inference for real-time agent interaction
3. **Quality**: Llama 3.1 70B provides excellent reasoning
4. **Tool Calling**: Native support for agent tool usage
5. **Cost Effective**: Competitive pricing for development

## Troubleshooting

### Common Issues

**"No module named 'groq'"**
```bash
pip install groq
# or
conda install -c conda-forge groq
```

**"API key not found"**
```bash
echo $GROQ_API_KEY  # Should show your key
export GROQ_API_KEY="your_key_here"
```

**"Rate limit exceeded"**
- Groq has generous rate limits
- Add retry logic (already implemented)
- Consider upgrading Groq plan if needed

### Getting Your API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Navigate to API Keys
4. Create new key
5. Copy and set as environment variable

## Next Steps

With Groq configured, you can:

1. **Run Core Tests**: `python test_core_functionality.py`
2. **Test Full System**: `python test_functional_agents.py`
3. **Start Universe**: `python src/main.py`
4. **Send Divine Messages**: Use the API endpoints

The agents will use Groq for all their reasoning, tool creation, security analysis, and performance evaluation - just like scaling up your original class examples to a complete AGI development environment!