# 运行命令：
# python bench.py --backend nanovllm
# python bench.py --backend vllm

import argparse
import os
import time
from random import randint, seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["vllm", "nanovllm"],
        default="vllm",
        help="Inference backend to benchmark.",
    )
    parser.add_argument(
        "--model",
        default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        help="Local model path.",
    )
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Lower than vLLM's default 0.9 to fit 8GB GPUs more reliably.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture.",
    )
    return parser.parse_args()


def load_backend(backend: str):
    if backend == "vllm":
        from vllm import LLM, SamplingParams
    else:
        from nanovllm import LLM, SamplingParams
    return LLM, SamplingParams


def main():
    args = parse_args()
    LLM, SamplingParams = load_backend(args.backend)

    seed(0)
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    min_input_len = min(100, args.max_input_len)
    min_output_len = min(100, args.max_output_len)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(min_input_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(min_output_len, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]
    if args.backend == "vllm":
        prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(
        f"Backend: {args.backend}, Total: {total_tokens}tok, "
        f"Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
