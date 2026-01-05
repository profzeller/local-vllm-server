# Local vLLM Server

High-throughput LLM inference server using vLLM with OpenAI-compatible API.

Optimized for 16GB VRAM GPUs with 2-4x faster inference than Ollama for batch/concurrent workloads.

## Quick Start

```bash
# Clone and configure
git clone https://github.com/profzeller/local-vllm-server.git
cd local-vllm-server
cp .env.example .env

# Edit .env to select your model
nano .env

# Start the server
docker compose up -d

# Watch logs (first start downloads model)
docker logs -f vllm
```

## Model Configuration

All configuration is in `.env`:

```bash
# HuggingFace model ID
VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3

# Model name exposed via API
VLLM_SERVED_NAME=mistral-7b

# Maximum context length
VLLM_MAX_MODEL_LEN=8192

# GPU memory utilization (0.0-1.0)
VLLM_GPU_MEMORY_UTIL=0.90
```

### Changing Models

```bash
# Edit the model
nano .env

# Restart to apply
docker compose down
docker compose up -d
```

## Recommended Models for 16GB VRAM

| Model | Size | Context | Speed | Notes |
|-------|------|---------|-------|-------|
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | 8K | Fast | Great all-around |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | 32K | Fast | Multilingual, long context |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | 8K | Very fast | Smaller but capable |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | 4K | Very fast | Efficient reasoning |

### For 24GB+ VRAM

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen2.5-14B-Instruct` | 14B | Best quality |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Good balance |

## Why vLLM over Ollama?

| Feature | vLLM | Ollama |
|---------|------|--------|
| **Throughput** | 2-4x faster | Baseline |
| **Batching** | Continuous batching | Limited |
| **Concurrent requests** | Excellent | Limited |
| **Memory efficiency** | PagedAttention | Standard |
| **Best for** | Production, batch generation | Dev, interactive |

**Use vLLM when:** Generating lots of content (social posts, articles, etc.)
**Use Ollama when:** Interactive development, simple testing

## API Usage

vLLM provides an **OpenAI-compatible API** on port 8000.

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [
      {"role": "user", "content": "Write 3 benefits of morning meditation."}
    ]
  }'
```

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="mistral-7b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Using Gated Models (Llama, etc.)

Some models require HuggingFace authentication:

1. Create account at [huggingface.co](https://huggingface.co)
2. Accept the model's license agreement
3. Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Add to `.env`:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxx
   ```

## Running Larger Models with Quantization

For models that don't fit in VRAM, use quantization:

```bash
# In .env
VLLM_MODEL=TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ
VLLM_QUANTIZATION=awq
VLLM_MAX_MODEL_LEN=4096
```

## Troubleshooting

### Out of Memory

1. Reduce `VLLM_MAX_MODEL_LEN` (e.g., 4096 or 2048)
2. Reduce `VLLM_GPU_MEMORY_UTIL` (e.g., 0.85)
3. Use a smaller model
4. Use a quantized model (AWQ/GPTQ)

### Container Crashes on Startup

```bash
# Check logs
docker logs vllm

# Common issues:
# - Model too large for VRAM → use smaller model
# - HF_TOKEN missing for gated model
# - Network issues downloading model
```

### Slow First Request

Normal - vLLM compiles CUDA kernels on first inference. Subsequent requests are fast.

## Configuration Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_MODEL` | HuggingFace model ID | mistralai/Mistral-7B-Instruct-v0.3 |
| `VLLM_SERVED_NAME` | Name in API responses | mistral-7b |
| `VLLM_MAX_MODEL_LEN` | Max context length | 8192 |
| `VLLM_GPU_MEMORY_UTIL` | VRAM usage (0.0-1.0) | 0.90 |
| `VLLM_DTYPE` | Data type | auto |
| `VLLM_QUANTIZATION` | Quantization method | (none) |
| `HF_TOKEN` | HuggingFace token | (none) |

## Management

```bash
# View logs
docker compose logs -f

# Check GPU usage
nvidia-smi

# Restart
docker compose restart

# Stop
docker compose down

# Update vLLM
docker compose pull
docker compose up -d
```

## Port

- `8000` - OpenAI-compatible API

## Files

```
local-vllm-server/
├── docker-compose.yml    # Main configuration (uses .env)
├── .env.example          # Configuration template
├── .env                  # Your config (not in git)
└── README.md
```

## License

MIT
