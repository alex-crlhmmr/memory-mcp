# Memory MCP Server

A Python MCP server that gives Claude Code persistent memory across conversations. Extracts preferences, code patterns, and subtle signals from conversations, embeds them with Qwen3-Embedding-8B (load-on-demand GPU), stores in LanceDB, and retrieves via vector similarity search.

## Tools

| Tool | Description | API Cost |
|------|-------------|----------|
| `save_conversation_memory(conversation_text)` | Extract + embed + store observations from a conversation | ~$0.01-0.05 |
| `recall_memories(query, limit, category)` | Search past observations + return user profile | Free |
| `update_memory(observation_id, profile_key, new_value, delete)` | Correct/remove specific memories | Free |

## Setup

### 1. Create conda environment

```bash
conda create -n memory_mcp python=3.12 -y
conda activate memory_mcp
```

### 2. Install Jetson PyTorch (Jetson only)

If you're on a Jetson device, install the appropriate PyTorch wheel before the package:

```bash
# Check https://developer.nvidia.com/embedded/downloads for your Jetpack version
pip install <jetson-pytorch-wheel-url>
```

On other platforms, the default PyTorch from PyPI will work.

### 3. Install the package

```bash
cd /path/to/memory_mcp
pip install -e ".[dev]"
```

### 4. Download the embedding model

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Embedding-8B', local_dir='models/Qwen3-Embedding-8B')
"
```

This downloads ~15GB of model weights. The model loads to CPU RAM on first use (~30s), then subsequent calls move it to GPU briefly for encoding (~2-3s) and offload back to CPU.

### 5. Set up Anthropic API key

The extraction step calls Claude Sonnet to analyze conversations. You need an API key from https://console.anthropic.com/settings/keys.

```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
```

### 6. Register with Claude Code

```bash
claude mcp add memory -- /path/to/miniforge3/envs/memory_mcp/bin/python -m memory_mcp.server
```

Replace the Python path with your actual conda env path. Find it with:

```bash
conda activate memory_mcp
which python
```

### 7. Add auto-save instruction

Add the following to `~/.claude/CLAUDE.md` (create the file if it doesn't exist):

```
At the start of every conversation, call recall_memories with a query relevant to the user's first message to load context about their preferences and past decisions.

When the user asks how they would approach something, what they'd prefer, or asks you to consult memory, call recall_memories with a relevant query and use the returned profile + memories to give a personalized answer.

At the end of every conversation, before closing, call save_conversation_memory with the full conversation text.
```

### 8. Verify

Restart Claude Code. The `save_conversation_memory`, `recall_memories`, and `update_memory` tools should be available. Have a conversation, let it save at the end, then start a new conversation and check that memories are recalled.

## Configuration

Environment variables (prefix `MEMORY_MCP_` or set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required. Anthropic API key for extraction |
| `MEMORY_MCP_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` | HuggingFace model ID |
| `MEMORY_MCP_EMBEDDING_DIM` | `1024` | Embedding vector dimensions |
| `MEMORY_MCP_GPU_MEMORY_THRESHOLD_GB` | `4.0` | Min free GPU memory to use CUDA |
| `MEMORY_MCP_EXTRACTION_MODEL` | `claude-sonnet-4-20250514` | Claude model for extraction |
| `MEMORY_MCP_DEFAULT_RECALL_LIMIT` | `20` | Default number of memories returned |
| `MEMORY_MCP_RELEVANCE_THRESHOLD` | `0.3` | Min similarity score for recall results |

## No GPU? No problem

The server works without a GPU — the embedding model runs on CPU automatically. The 8B model will be slow on CPU though (~30-60s per embed call). If you're CPU-only, consider using a smaller embedding model by setting:

```bash
MEMORY_MCP_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
MEMORY_MCP_EMBEDDING_DIM=512
```

Then download the smaller model (~1.2GB instead of 15GB):

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/Qwen3-Embedding-0.6B')
"
```

## How it works

**Save flow:**
1. Claude Sonnet analyzes the conversation and extracts 5-30 observations across 7 categories (coding preferences, code patterns, technical context, decisions, subtle signals, communication patterns, project context)
2. Qwen3-Embedding-8B embeds all observations into 1024-dim vectors
3. Observations + vectors stored in LanceDB
4. High-confidence preferences merged into the user profile

**Recall flow:**
1. Query embedded with Qwen3
2. Vector similarity search in LanceDB
3. Returns user profile + ranked relevant memories

**GPU management:**
- Model stays in CPU RAM between calls
- On embed: check GPU free memory → move to CUDA if >4GB free, else CPU fallback → encode → offload to CPU → `torch.cuda.empty_cache()`
- RL training or other GPU workloads are never disrupted

## Running tests

```bash
conda activate memory_mcp
pytest tests/ -v
```

## Project structure

```
memory_mcp/
├── pyproject.toml
├── .env                       # API key (gitignored)
├── src/memory_mcp/
│   ├── server.py              # MCP server entry point (FastMCP, stdio)
│   ├── config.py              # Settings with env var overrides
│   ├── models.py              # Pydantic data models
│   ├── tools/
│   │   ├── save.py            # save_conversation_memory
│   │   ├── recall.py          # recall_memories
│   │   └── update.py          # update_memory
│   ├── extraction/
│   │   └── extractor.py       # Claude API conversation analysis
│   ├── embeddings/
│   │   └── manager.py         # Load-on-demand Qwen3-8B embedding manager
│   └── storage/
│       ├── lance_store.py     # LanceDB vector operations
│       └── profile.py         # Aggregated user profile (JSON)
├── data/                      # LanceDB data + profile (gitignored)
├── models/                    # Cached model weights (gitignored)
└── tests/
```
