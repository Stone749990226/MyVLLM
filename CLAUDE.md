# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines) built from scratch with PyTorch. It achieves comparable performance to vLLM through optimizations including prefix caching, tensor parallelism, CUDA graphs, and Triton kernels.

## Essential Commands

### Installation
```bash
pip install -e .                    # Editable install for development
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git  # From git
```

### Running Examples
```bash
python example.py                   # Basic usage example
python bench.py                     # Performance benchmark
```

### Model Download
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Architecture Overview

### Core Components

The codebase follows a layered architecture with clear separation of concerns:

1. **LLM API Layer** ([nanovllm/llm.py](nanovllm/llm.py))
   - Thin wrapper around LLMEngine
   - Public API: `LLM.generate(prompts, sampling_params)`

2. **Engine Layer** ([nanovllm/engine/](nanovllm/engine/))
   - **LLMEngine**: Orchestrates tokenization, scheduling, and model execution
   - **Scheduler**: Manages two-phase inference (prefill → decode)
   - **BlockManager**: Handles KV cache allocation with prefix caching
   - **ModelRunner**: Executes model inference with CUDA graphs
   - **Sequence**: Tracks per-request state and token IDs

3. **Model Layer** ([nanovllm/models/](nanovllm/models/))
   - Currently supports Qwen3 ([qwen3.py](nanovllm/models/qwen3.py))
   - Models implement `forward(input_ids, positions)` and `compute_logits()`

4. **Layers** ([nanovllm/layers/](nanovllm/layers/))
   - Custom implementations with Triton kernels
   - Attention, linear layers, sampler, embeddings, etc.

### Key Design Patterns

#### Two-Phase Inference Pattern

The scheduler operates in two distinct phases:

1. **Prefill Phase** ([scheduler.py:25-47](nanovllm/engine/scheduler.py#L25-L47))
   - Processes new requests from `waiting` queue
   - Computes KV cache for all prompt tokens
   - Leverages prefix caching to skip redundant computation
   - Returns `(sequences, is_prefill=True)`

2. **Decode Phase** ([scheduler.py:49-69](nanovllm/engine/scheduler.py#L49-L69))
   - Processes running sequences one token at a time
   - Uses CUDA graphs for batch sizes 1-512 (when `enforce_eager=False`)
   - Implements preemption when KV cache is full
   - Returns `(sequences, is_prefill=False)`

**Critical**: Prefill always takes priority. Decode only runs when `waiting` queue is empty.

#### KV Cache Block Management

The BlockManager ([block_manager.py](nanovllm/engine/block_manager.py)) implements a sophisticated caching strategy:

- **Block-based allocation**: KV cache divided into fixed-size blocks (default 256 tokens)
- **Prefix caching**: Uses xxhash to identify and reuse identical prefix blocks
- **Reference counting**: Multiple sequences can share cached blocks
- **Hash chaining**: Only consecutive prefix blocks from sequence start are cached

Key insight: `num_cached_tokens` tracks how many tokens were cache hits during allocation, reducing computation in prefill.

#### Context-Based Inference

Instead of passing metadata through function arguments, the codebase uses a global context pattern ([utils/context.py](nanovllm/utils/context.py)):

```python
# Set before model execution
set_context(is_prefill, cu_seqlens_q, slot_mapping, block_tables, ...)

# Access in layer forward passes
context = get_context()

# Clear after execution
reset_context()
```

This allows layers (attention, etc.) to access inference metadata without modifying signatures.

#### Tensor Parallelism via Multiprocessing

Tensor parallelism is implemented using `torch.multiprocessing` with spawn context ([model_runner.py:23-50](nanovllm/engine/model_runner.py#L23-L50)):

- Rank 0 is the coordinator, ranks 1+ are workers
- Workers run in infinite loop waiting for commands via shared memory
- Communication uses pickle serialization + multiprocessing Events
- All ranks participate in `dist.all_reduce` for tensor-parallel operations

#### CUDA Graph Optimization

For decode phase with batch sizes ≤512 ([model_runner.py:296-336](nanovllm/engine/model_runner.py#L296-L336)):

- Pre-captures graphs for batch sizes: [1, 2, 4, 8, 16, 32, ..., max_bs]
- Reuses graph pool to minimize memory overhead
- Replays appropriate graph based on actual batch size
- Falls back to eager mode for prefill or large batches

### Configuration

The Config dataclass ([config.py](nanovllm/config.py)) controls key parameters:

- `max_num_batched_tokens`: Maximum tokens per batch (default 16384)
- `max_num_seqs`: Maximum sequences per batch (default 512)
- `max_model_len`: Maximum sequence length (default 4096)
- `gpu_memory_utilization`: GPU memory fraction for KV cache (default 0.9)
- `tensor_parallel_size`: Number of GPUs for tensor parallelism (1-8)
- `enforce_eager`: Disable CUDA graphs (default False)
- `kvcache_block_size`: KV cache block size, must be multiple of 256 (default 256)

### Sequence State Management

Each Sequence ([sequence.py](nanovllm/engine/sequence.py)) tracks:

- `token_ids`: All tokens (prompt + generated)
- `num_prompt_tokens`: Length of original prompt
- `num_cached_tokens`: How many tokens hit prefix cache
- `block_table`: List of KV cache block IDs allocated to this sequence
- `status`: WAITING → RUNNING → FINISHED

Key properties:
- `num_completion_tokens`: Generated tokens count
- `num_blocks`: Total KV cache blocks needed
- `num_cached_blocks`: Blocks that were cache hits

### Scheduler Preemption

When KV cache is full during decode ([scheduler.py:52-60](nanovllm/engine/scheduler.py#L52-L60)):

1. Pop sequences from end of `running` queue
2. Deallocate their KV cache blocks
3. Move them back to front of `waiting` queue
4. Repeat until current sequence can proceed

This ensures forward progress while maximizing throughput.

## Code Style

### Type Hints
Use Python 3.10+ union syntax: `list[str] | list[list[int]]` instead of `Optional[...]`

### Imports
Organize in three groups: standard library, third-party, local imports (separated by blank lines)

### Naming
- Classes: PascalCase (`LLMEngine`, `BlockManager`)
- Functions/variables: snake_case (`add_request`, `num_tokens`)
- Private members: underscore prefix (`_allocate_block`)

### Error Handling
Use `assert` for internal invariants, let exceptions propagate for external errors

## Important Notes

- **Model path must be a directory**: Config validates `os.path.isdir(model)`
- **Block size must be multiple of 256**: Required for Triton kernel alignment
- **Greedy sampling not supported**: `temperature` must be > 1e-10
- **Tensor parallel size**: 1-8 GPUs supported
- **CUDA graphs**: Only used for decode phase with batch size ≤ 512

## Reference Locations

- Main generate API: [llm_engine.py:59](nanovllm/engine/llm_engine.py#L59)
- Scheduler logic: [scheduler.py:25](nanovllm/engine/scheduler.py#L25)
- Model execution: [model_runner.py:284](nanovllm/engine/model_runner.py#L284)
- KV cache allocation: [block_manager.py:59](nanovllm/engine/block_manager.py#L59)
- Prefix cache hashing: [block_manager.py:34](nanovllm/engine/block_manager.py#L34)
