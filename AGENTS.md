# AGENTS.md - nano-vLLM Development Guide

This document provides guidelines for AI agents working on the nano-vLLM codebase.

Python环境：source /home/stone/learn/pytorch/nano-vllm/.venv/bin/activate
## Project Overview

nano-vLLM is a lightweight vLLM implementation (~1,200 lines) built from scratch with PyTorch. It implements:
- LLM inference engine with batch scheduling
- KV cache management with prefix caching
- Tensor parallelism support
- CUDA graph optimization
- Triton kernel implementations

## Build/Lint/Test Commands

### Installation
```bash
pip install -e .                    # Editable install
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git  # From git
```

### Running Tests
```bash
pytest                              # Run all tests
pytest -v                           # Verbose output
pytest tests/test_specific.py       # Run specific test file
pytest tests/ -k "test_name"        # Run tests matching pattern
```

### Type Checking (if configured)
```bash
mypy nanovllm/                      # Type check entire package
mypy --strict nanovllm/             # Strict mode
```

### Linting (if configured)
```bash
ruff check nanovllm/                # Lint with ruff
ruff check --fix nanovllm/          # Auto-fix issues
```

## Code Style Guidelines

### Imports

Organize imports in three groups separated by blank lines:
1. Standard library imports
2. Third-party imports
3. Local/nanovllm imports

Example:
```python
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoConfig

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
```

### Type Hints

Use modern Python 3.10+ union syntax:
```python
def generate(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
) -> list[str]:
```

Prefer explicit type annotations for function parameters and return values. Use `| None` instead of `Optional[...]`.

### Naming Conventions

| Element | Convention | Examples |
|---------|------------|----------|
| Classes | PascalCase | `LLMEngine`, `SequenceStatus`, `BlockManager` |
| Functions | snake_case | `add_request`, `prepare_prefill`, `allocate_kv_cache` |
| Variables | snake_case | `num_tokens`, `block_table`, `is_prefill` |
| Constants | UPPER_SNAKE_CASE | `MAX_BATCH_SIZE` |
| Private members | underscore prefix | `_warmup_model`, `_internal_state` |

### Error Handling

- Use `assert` for internal invariants and parameter validation
- Use descriptive assert messages for complex conditions
- Let exceptions propagate for external errors (e.g., file not found, CUDA errors)
- Use `raise ValueError(...)` for invalid argument combinations

```python
def __post_init__(self):
    assert os.path.isdir(self.model), f"Model path not found: {self.model}"
    assert self.kvcache_block_size % 256 == 0
    assert 1 <= self.tensor_parallel_size <= 8
```

### Docstrings

Use Google-style docstrings for public methods:

```python
def prepare_prefill(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare input tensors for prefill phase.

    Args:
        seqs: List of sequences to process in prefill

    Returns:
        Tuple of (input_ids, positions) tensors ready for model inference
    """
```

### Code Structure

#### Directory Layout

```
nanovllm/
├── __init__.py           # Public API exports
├── llm.py                # Main LLM class (thin wrapper)
├── config.py             # Configuration dataclass
├── sampling_params.py    # Sampling parameters
├── engine/
│   ├── llm_engine.py     # Core engine implementation
│   ├── scheduler.py      # Batching scheduler
│   ├── model_runner.py   # Model execution & CUDA graphs
│   ├── sequence.py       # Sequence state management
│   └── block_manager.py  # KV cache block allocation
├── layers/
│   ├── attention.py      # Flash attention with Triton kernels
│   ├── linear.py         # Linear layers with tensor parallelism
│   ├── sampler.py        # Token sampling logic
│   └── ...
├── models/
│   └── qwen3.py          # Qwen3 model implementation
└── utils/
    ├── context.py        # Inference context management
    └── loader.py         # Model weight loading
```

#### File Organization

- Keep files under 300 lines when possible
- Group related functionality in modules
- Use dataclasses for configuration objects
- Use enums for status/flag values

### Performance Considerations

- Use `@torch.inference_mode()` for inference code
- Pre-allocate tensors with `pin_memory=True` for GPU transfers
- Use CUDA graphs for batch sizes 1-512 when `enforce_eager=False`
- Consider memory layout for Triton kernels (contiguous strides)
- Use `torch.multiprocessing` for tensor parallelism

### Tensor Parallelism

When implementing tensor-parallel features:
- Use `torch.distributed` for cross-GPU communication
- Follow pattern in `model_runner.py` rank-based initialization
- Use `dist.barrier()` for synchronization points
- Handle rank-0 as coordinator with shared memory

## Key Patterns

### Context-based Inference

```python
# Set context before model run
set_context(
    is_prefill=True,
    cu_seqlens_q=cu_seqlens_q,
    slot_mapping=slot_mapping,
    block_tables=block_tables,
)
# Access in layer forward passes
context = get_context()
```

### Scheduler Pattern

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # 1. Try prefill for waiting sequences
    # 2. If no prefill, do decode for running sequences
    # 3. Return (sequences, is_prefill_flag)
```

### Triton Kernel Pattern

```python
@triton.jit
def kernel_function(ptr, stride, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0)
    # kernel implementation
```

## Testing Guidelines

### Test Organization

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use pytest framework
- Focus on:
  - Scheduler correctness (prefill/decode ordering)
  - Block allocation/deallocation
  - Sequence state transitions
  - Integration with model inference

### Test Examples

```python
def test_scheduler_prefill_order():
    scheduler = Scheduler(config)
    # Test prefill scheduling logic

def test_sequence_append():
    seq = Sequence([1, 2, 3])
    seq.append_token(4)
    assert len(seq) == 4
```

## Common Tasks

### Adding a New Model

1. Create `nanovllm/models/<model_name>.py`
2. Implement `forward()` with `input_ids` and `positions`
3. Implement `compute_logits()` returning logits tensor
4. Add to `ModelRunner.__init__()` initialization

### Adding a New Layer

1. Create `nanovllm/layers/<layer_name>.py`
2. Inherit from `torch.nn.Module`
3. Implement `forward()` method
4. Add layer creation in model implementation

### Modifying Scheduler

1. Understand the two-phase pattern (prefill → decode)
2. Maintain `waiting` and `running` deques correctly
3. Handle block allocation failures with preemption
4. Update `SequenceStatus` appropriately

## Reference Files

- Main API: `nanovllm/engine/llm_engine.py:42` (generate method)
- Scheduling: `nanovllm/engine/scheduler.py:25`
- Model execution: `nanovllm/engine/model_runner.py:284`
- Attention: `nanovllm/layers/attention.py:43`
