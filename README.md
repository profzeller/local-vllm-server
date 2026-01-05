# Local vLLM Server

High-throughput LLM inference server using vLLM on a local GPU.

Optimized for 16GB VRAM GPUs with 2-4x faster inference than Ollama for batch/concurrent workloads.

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

## Requirements

- NVIDIA GPU with 16GB VRAM
- Ubuntu Server 22.04+ with Docker
- NVIDIA Container Toolkit
- HuggingFace account (for gated models like Llama)

## VRAM Usage

| Model | VRAM | Speed |
|-------|------|-------|
| Qwen 2.5 14B | ~14GB | Good |
| Llama 3.1 8B | ~10GB | Fast |
| Mistral 7B | ~8GB | Fastest |

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/profzeller/local-vllm-server.git
cd local-vllm-server
```

### 2. (Optional) Set HuggingFace token

Required for gated models like Llama:

```bash
# Get token from https://huggingface.co/settings/tokens
echo "HF_TOKEN=hf_your_token_here" > .env
```

### 3. Start the server

```bash
# Default: Qwen 2.5 14B (recommended for 16GB)
docker compose up -d

# Alternative: Llama 3.1 8B (faster)
docker compose -f docker-compose.llama.yml up -d

# Alternative: Mistral 7B (fastest)
docker compose -f docker-compose.mistral.yml up -d
```

First start downloads the model (~15-30GB) and takes a few minutes.

### 4. Verify it's running

```bash
# Check health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models
```

## API Usage

vLLM provides an **OpenAI-compatible API** on port 8000.

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-14b",
    "messages": [
      {"role": "system", "content": "You are a wellness content expert."},
      {"role": "user", "content": "Write 3 benefits of morning meditation."}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### Text Completion

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-14b",
    "prompt": "Write a wellness tip about hydration:",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Python Example

```python
from openai import OpenAI

# Point to local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM doesn't require auth
)

response = client.chat.completions.create(
    model="qwen2.5-14b",
    messages=[
        {"role": "system", "content": "You are a wellness content expert."},
        {"role": "user", "content": "Write a short Instagram caption about mindfulness."}
    ],
    temperature=0.7,
    max_tokens=300
)

print(response.choices[0].message.content)
```

### Batch Generation (vLLM's strength)

```python
from openai import OpenAI
import asyncio

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

prompts = [
    "Write an Instagram caption about morning routines",
    "Write a Twitter post about meditation benefits",
    "Write a LinkedIn post about work-life balance",
    "Write a TikTok script about healthy habits",
]

# vLLM handles these concurrently with continuous batching
import concurrent.futures

def generate(prompt):
    return client.chat.completions.create(
        model="qwen2.5-14b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

# Run all in parallel - vLLM batches efficiently
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(generate, prompts))

for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt[:50]}...")
    print(f"Response: {result.choices[0].message.content[:100]}...")
    print()
```

## Model Configuration

### Available Compose Files

| File | Model | VRAM | Use Case |
|------|-------|------|----------|
| `docker-compose.yml` | Qwen 2.5 14B | ~14GB | Best quality |
| `docker-compose.llama.yml` | Llama 3.1 8B | ~10GB | Good balance |
| `docker-compose.mistral.yml` | Mistral 7B | ~8GB | Fastest |

### Custom Model

Edit `docker-compose.yml` and change the `--model` parameter:

```yaml
command: >
  --model your-org/your-model
  --dtype auto
  --max-model-len 4096
  --gpu-memory-utilization 0.90
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | HuggingFace model ID | Qwen/Qwen2.5-14B-Instruct |
| `--dtype` | Data type (auto, float16, bfloat16) | auto |
| `--max-model-len` | Max context length | 4096 |
| `--gpu-memory-utilization` | VRAM usage (0.0-1.0) | 0.90 |
| `--served-model-name` | Name in API responses | model name |

## Integrating with Your App

### If your app uses Ollama

vLLM's OpenAI-compatible API means minimal changes:

```typescript
// Before (Ollama)
const response = await fetch('http://ollama-server:11434/api/chat', {
  method: 'POST',
  body: JSON.stringify({
    model: 'qwen2.5:14b',
    messages: [...]
  })
});

// After (vLLM) - use OpenAI format
const response = await fetch('http://vllm-server:8000/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'qwen2.5-14b',
    messages: [...]
  })
});
```

### Create a vLLM provider

Add a new provider in your app that uses the OpenAI SDK pointed at vLLM:

```typescript
import OpenAI from 'openai';

const vllm = new OpenAI({
  baseURL: 'http://vllm-server:8000/v1',
  apiKey: 'not-needed',
});
```

## Network Configuration

Default port: **8000**

Access from other machines:
```
http://<server-ip>:8000
```

### Firewall

```bash
sudo ufw allow from 209.114.126.0/24 to any port 8000 proto tcp
sudo ufw allow from 24.225.29.131 to any port 8000 proto tcp
```

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

# Switch models
docker compose down
docker compose -f docker-compose.llama.yml up -d

# Update vLLM
docker compose pull
docker compose up -d
```

## Performance Comparison

Benchmark on RTX 4090 Laptop (16GB), generating 500-token responses:

| Metric | vLLM (Qwen 14B) | Ollama (Qwen 14B) |
|--------|-----------------|-------------------|
| Single request | ~3s | ~4s |
| 4 concurrent | ~4s total | ~16s total |
| 10 concurrent | ~8s total | ~40s total |
| Tokens/sec | ~150 | ~80 |

**vLLM shines with concurrent/batch requests.**

## Troubleshooting

### Out of memory

```bash
# Use a smaller model
docker compose -f docker-compose.mistral.yml up -d

# Or reduce context length in docker-compose.yml:
--max-model-len 2048

# Or reduce memory utilization:
--gpu-memory-utilization 0.80
```

### Model download fails

```bash
# Check HuggingFace token
cat .env

# For gated models, accept license at:
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

# Check logs
docker compose logs -f
```

### Slow first request

Normal - vLLM compiles CUDA kernels on first inference. Subsequent requests are fast.

### Connection refused

```bash
# Check if container is running
docker ps

# Check health (may take 2+ min on first start)
docker compose logs -f | grep -i "started"
```

## License

MIT License - Use freely for personal and commercial projects.

## Credits

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [Qwen](https://github.com/QwenLM/Qwen2.5) - Excellent open-source LLM
