# Memory MCP Server

A Python MCP server that gives Claude Code persistent memory across conversations. Extracts preferences, code patterns, and decisions from conversations using Claude, embeds them with sentence-transformers, stores in LanceDB, and retrieves via vector similarity search.

## Tools

| Tool | Description | API Cost |
|------|-------------|----------|
| `save_conversation_memory(conversation_text)` | Extract + embed + store observations from a conversation | ~$0.01-0.05 |
| `recall_memories(query, limit, category)` | Search past observations + return user profile | Free |
| `update_memory(observation_id, profile_key, new_value, delete)` | Correct/remove specific memories | Free |

**Categories** extracted: `coding_preference`, `communication_pattern`, `code_pattern`, `technical_context`, `decision`, `subtle_signal`, `project_context`

## Setup

### 1. Create conda environment

```bash
conda create -n memory_mcp python=3.12 -y
conda activate memory_mcp
```

### 2. Install PyTorch

The embedding model requires PyTorch. Install the version that matches your hardware **before** installing the package:

- **CPU (default):** No extra steps needed — `pip install` in step 3 will pull in a CPU-compatible PyTorch automatically.
- **NVIDIA GPU:** Install the CUDA-enabled build from https://pytorch.org/get-started/locally/ for faster embedding.
- **Jetson / other devices:** Install the appropriate wheel for your platform first (e.g. check https://developer.nvidia.com/embedded/downloads for Jetson).

### 3. Install the package

```bash
cd /path/to/memory_mcp
pip install -e ".[dev]"
```

The default embedding model (`all-MiniLM-L6-v2`) downloads automatically on first use (~80MB). No manual model download needed.

### 4. Set up Anthropic API key

The extraction step calls Claude Sonnet to analyze conversations. You need an API key from https://console.anthropic.com/settings/keys.

```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
```

### 5. Register with Claude Code

```bash
claude mcp add memory -- /path/to/miniforge3/envs/memory_mcp/bin/python -m memory_mcp.server
```

Replace the Python path with your actual conda env path. Find it with:

```bash
conda activate memory_mcp
which python
```

### 6. Add auto-save instruction

Add the following to `~/.claude/CLAUDE.md` (create the file if it doesn't exist):

```
At the start of every conversation, call recall_memories with a query relevant to the user's first message to load context about their preferences and past decisions.

When the user asks how they would approach something, what they'd prefer, or asks you to consult memory, call recall_memories with a relevant query and use the returned profile + memories to give a personalized answer.

During a conversation, if you hit a decision point (e.g. choosing an architecture, picking a pattern, explaining a concept) where the user's past preferences or decisions could change the answer, call recall_memories with a focused query. Don't call it speculatively — only when you have a concrete reason to think stored context would shift your approach. Also, if a long stretch of the conversation has passed without a recall, consider calling it with a query relevant to the current topic — context from earlier sessions may have become relevant as the problem evolved.

At the end of every conversation, before closing, call save_conversation_memory with the full conversation text.
```

### 7. Verify

Restart Claude Code. The `save_conversation_memory`, `recall_memories`, and `update_memory` tools should be available. Have a conversation, let it save at the end, then start a new conversation and check that memories are recalled.

## Automatic saving with SessionEnd hook (optional)

Instead of relying on Claude to call `save_conversation_memory` before closing, you can use a `SessionEnd` hook to automatically save every conversation when you `/exit`.

The `memory-save` CLI command is installed alongside the package. It reads the session transcript directly and calls the save pipeline without going through MCP.

Add this to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "memory-save"
          }
        ]
      }
    ]
  }
}
```

If you already have other settings in that file, merge the `hooks` key into your existing config.

**How it works:** When you `/exit`, Claude Code pipes a JSON object with `transcript_path` to stdin. `memory-save` reads the `.jsonl` transcript, extracts user/assistant messages, and runs the full save pipeline (extraction via Claude API + embedding + LanceDB storage). This runs after the session ends, so there's no delay.

**Manual test:**

```bash
echo '{"transcript_path":"/path/to/transcript.jsonl"}' | memory-save
```

Transcript files live in `~/.claude/projects/` under directories named after your project path.

## No GPU? No problem

The embedding model runs on CPU automatically if no GPU is available. The default model (`all-MiniLM-L6-v2`) is small and fast — embedding takes under a second on CPU.

When a CUDA GPU is detected, it's used automatically for faster batch embedding. The `MEMORY_MCP_EMBEDDING_DEVICE` setting controls this:

- `auto` (default): use GPU if available, fall back to CPU
- `cpu`: force CPU
- `cuda`: force GPU

## Configuration

Environment variables (prefix `MEMORY_MCP_` or set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required. Anthropic API key for extraction |
| `MEMORY_MCP_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model ID |
| `MEMORY_MCP_EMBEDDING_DIM` | `384` | Embedding vector dimensions (must match model) |
| `MEMORY_MCP_EMBEDDING_DEVICE` | `auto` | Device: `auto`, `cpu`, or `cuda` |
| `MEMORY_MCP_GPU_MEMORY_THRESHOLD_GB` | `4.0` | Min free GPU memory to use CUDA |
| `MEMORY_MCP_EXTRACTION_MODEL` | `claude-sonnet-4-20250514` | Claude model for extraction |
| `MEMORY_MCP_DATA_DIR` | `data/` | Data directory |
| `MEMORY_MCP_MODELS_DIR` | `models/` | Model cache directory |
| `MEMORY_MCP_LANCE_DB_PATH` | `data/lancedb` | LanceDB storage path |
| `MEMORY_MCP_PROFILE_PATH` | `data/user_profile.json` | User profile JSON path |
| `MEMORY_MCP_DEFAULT_RECALL_LIMIT` | `20` | Default number of memories returned |
| `MEMORY_MCP_RELEVANCE_THRESHOLD` | `0.3` | Min similarity score for recall results |

## How it works

**Save flow:**
1. Claude Sonnet analyzes the conversation via streaming extraction, emitting observations incrementally
2. Observations are embedded into 384-dim vectors using sentence-transformers
3. Observations + vectors stored in LanceDB
4. High-confidence preferences (>=0.7) merged into the user profile

**Recall flow:**
1. Query embedded with the same model
2. Vector similarity search in LanceDB (cosine distance)
3. Returns user profile + ranked relevant memories above the relevance threshold

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
│   ├── save_cli.py            # CLI entry point for SessionEnd hook
│   ├── config.py              # Settings with env var overrides
│   ├── models.py              # Pydantic data models
│   ├── tools/
│   │   ├── save.py            # save_conversation_memory
│   │   ├── recall.py          # recall_memories
│   │   └── update.py          # update_memory
│   ├── extraction/
│   │   └── extractor.py       # Claude API streaming conversation analysis
│   ├── embeddings/
│   │   └── manager.py         # Sentence-transformers embedding manager
│   └── storage/
│       ├── lance_store.py     # LanceDB vector operations
│       └── profile.py         # Aggregated user profile (JSON)
├── data/                      # LanceDB data + profile (gitignored)
├── models/                    # Cached model weights (gitignored)
└── tests/
```
