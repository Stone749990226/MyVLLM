# 运行命令：
# python bench.py
# python bench.py --backend nanovllm
# python bench.py --backend vllm

import argparse
import atexit
import gc
import math
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from random import randint, seed
from time import perf_counter
from zoneinfo import ZoneInfo


BEIJING_TZ = ZoneInfo("Asia/Shanghai")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["all", "vllm", "nanovllm"],
        default="all",
        help="Inference backend to benchmark. 'all' runs nanovllm then vllm.",
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
    parser.add_argument(
        "--log-dir",
        default="log",
        help="Directory used to store benchmark logs.",
    )
    return parser.parse_args()


def load_backend(backend: str):
    if backend == "vllm":
        from vllm import LLM, SamplingParams
    else:
        from nanovllm import LLM, SamplingParams
    return LLM, SamplingParams


def compute_percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    assert 0 <= q <= 100
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * q / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_metrics(
    ttfts: list[float],
    tpots: list[float],
    total_tokens: int,
    wall_time: float,
) -> dict[str, float | int | None]:
    assert ttfts, "TTFT metrics must not be empty."
    return {
        "ttft_avg": sum(ttfts) / len(ttfts),
        "ttft_p50": compute_percentile(ttfts, 50),
        "ttft_p95": compute_percentile(ttfts, 95),
        "tpot_avg": sum(tpots) / len(tpots) if tpots else None,
        "total_tokens": total_tokens,
        "wall_time": wall_time,
        "throughput": total_tokens / wall_time,
    }


def format_ms(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 1000:.2f}ms"


def format_metrics(
    backend: str,
    num_seqs: int,
    metrics: dict[str, float | int | None],
) -> str:
    total_tokens = metrics["total_tokens"]
    wall_time = metrics["wall_time"]
    throughput = metrics["throughput"]
    ttft_avg = metrics["ttft_avg"]
    ttft_p50 = metrics["ttft_p50"]
    ttft_p95 = metrics["ttft_p95"]
    tpot_avg = metrics["tpot_avg"]
    return "\n".join(
        [
            (
                f"Backend: {backend}, Requests: {num_seqs}, Output Tokens: {total_tokens}tok, "
                f"Time: {wall_time:.2f}s, Throughput: {throughput:.2f}tok/s"
            ),
            (
                f"TTFT: avg {format_ms(ttft_avg)}, "
                f"p50 {format_ms(ttft_p50)}, p95 {format_ms(ttft_p95)}"
            ),
            f"TPOT: avg {format_ms(tpot_avg)}",
        ]
    )


def build_request_specs(args) -> tuple[list[list[int]], list[int]]:
    min_input_len = min(100, args.max_input_len)
    min_output_len = min(100, args.max_output_len)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(min_input_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    max_tokens = [
        randint(min_output_len, args.max_output_len) for _ in range(args.num_seqs)
    ]
    return prompt_token_ids, max_tokens


def build_sampling_params(max_tokens: list[int], SamplingParams):
    return [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=token_count,
        )
        for token_count in max_tokens
    ]


def benchmark_vllm(llm, prompt_token_ids: list[list[int]], sampling_params):
    prompts = [dict(prompt_token_ids=prompt) for prompt in prompt_token_ids]
    arrival_times = {}
    first_token_times = {}
    finished_times = {}
    output_lengths = {}

    start = perf_counter()
    for prompt, sampling_param in zip(prompts, sampling_params):
        arrival_time = perf_counter()
        request_id = llm._add_request(prompt, sampling_param)
        arrival_times[request_id] = arrival_time

    while llm.llm_engine.has_unfinished_requests():
        step_outputs = llm.llm_engine.step()
        step_end = perf_counter()
        for output in step_outputs:
            assert len(output.outputs) == 1, "bench.py expects a single completion per request."
            completion = output.outputs[0]
            if output.request_id not in first_token_times and completion.token_ids:
                first_token_times[output.request_id] = step_end
            if output.finished and output.request_id not in finished_times:
                finished_times[output.request_id] = step_end
                output_lengths[output.request_id] = len(completion.token_ids)

    wall_time = perf_counter() - start

    ttfts = []
    tpots = []
    total_tokens = 0
    for request_id, arrival_time in arrival_times.items():
        first_token_time = first_token_times.get(request_id)
        finished_time = finished_times.get(request_id)
        output_tokens = output_lengths.get(request_id)
        assert first_token_time is not None, f"Missing first token time for request {request_id}."
        assert finished_time is not None, f"Missing finished time for request {request_id}."
        assert output_tokens is not None, f"Missing output length for request {request_id}."

        total_tokens += output_tokens
        ttfts.append(first_token_time - arrival_time)
        if output_tokens > 1:
            tpots.append((finished_time - first_token_time) / (output_tokens - 1))

    return summarize_metrics(ttfts, tpots, total_tokens, wall_time)


def benchmark_nanovllm(llm, prompt_token_ids: list[list[int]], sampling_params):
    arrival_times = {}
    first_token_times = {}
    finished_times = {}
    seq_refs = {}

    start = perf_counter()
    for prompt, sampling_param in zip(prompt_token_ids, sampling_params):
        arrival_time = perf_counter()
        llm.add_request(prompt, sampling_param)
        seq = llm.scheduler.waiting[-1]
        arrival_times[seq.seq_id] = arrival_time
        seq_refs[seq.seq_id] = seq

    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        step_end = perf_counter()

        for seq in seqs:
            if seq.seq_id not in first_token_times and seq.num_completion_tokens == 0:
                first_token_times[seq.seq_id] = step_end

        llm.scheduler.postprocess(seqs, token_ids)

        for seq in seqs:
            if seq.is_finished and seq.seq_id not in finished_times:
                finished_times[seq.seq_id] = step_end

    wall_time = perf_counter() - start

    ttfts = []
    tpots = []
    total_tokens = 0
    for seq_id, seq in seq_refs.items():
        first_token_time = first_token_times.get(seq_id)
        finished_time = finished_times.get(seq_id)
        assert first_token_time is not None, f"Missing first token time for sequence {seq_id}."
        assert finished_time is not None, f"Missing finished time for sequence {seq_id}."

        output_tokens = len(seq.completion_token_ids)
        total_tokens += output_tokens
        ttfts.append(first_token_time - arrival_times[seq_id])
        if output_tokens > 1:
            tpots.append((finished_time - first_token_time) / (output_tokens - 1))

    return summarize_metrics(ttfts, tpots, total_tokens, wall_time)


def benchmark(backend: str, llm, prompt_token_ids: list[list[int]], sampling_params):
    if backend == "vllm":
        return benchmark_vllm(llm, prompt_token_ids, sampling_params)
    return benchmark_nanovllm(llm, prompt_token_ids, sampling_params)


def get_run_timestamp(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now(BEIJING_TZ)
    return now.astimezone(BEIJING_TZ).strftime("%Y-%m-%d-%H-%M-%S")


def resolve_backends(backend: str) -> list[str]:
    if backend == "all":
        return ["nanovllm", "vllm"]
    return [backend]


def build_log_paths(log_dir: str | Path, timestamp: str, backends: list[str]):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    backend_logs = {
        backend: log_dir / f"{timestamp}-{backend}.log" for backend in backends
    }
    summary_log = log_dir / f"{timestamp}-summary.log"
    return backend_logs, summary_log


def cleanup_llm(llm):
    if llm is None:
        return
    try:
        if hasattr(llm, "exit"):
            atexit.unregister(llm.exit)
            llm.exit()
    except Exception:
        traceback.print_exc()
    finally:
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()


def run_backend(args, backend: str, prompt_token_ids: list[list[int]], max_tokens: list[int]):
    LLM, SamplingParams = load_backend(backend)
    sampling_params = build_sampling_params(max_tokens, SamplingParams)
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    metrics = benchmark(backend, llm, prompt_token_ids, sampling_params)
    return format_metrics(backend, args.num_seqs, metrics), metrics, llm


def execute_backend_with_logging(
    args,
    backend: str,
    prompt_token_ids: list[list[int]],
    max_tokens: list[int],
    log_path: Path,
):
    llm = None
    started_at = datetime.now(BEIJING_TZ)
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = Tee(sys.stdout, log_file)
        tee_stderr = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print(f"[{started_at.isoformat()}] Starting benchmark for {backend}")
            try:
                formatted_metrics, metrics, llm = run_backend(
                    args, backend, prompt_token_ids, max_tokens
                )
                print(formatted_metrics)
                status = "success"
                error = None
            except Exception as exc:
                traceback.print_exc()
                formatted_metrics = None
                metrics = None
                status = "failed"
                error = str(exc)
            finally:
                cleanup_llm(llm)
                finished_at = datetime.now(BEIJING_TZ)
                print(f"[{finished_at.isoformat()}] Finished benchmark for {backend}")

    return {
        "backend": backend,
        "status": status,
        "error": error,
        "formatted_metrics": formatted_metrics,
        "metrics": metrics,
        "log_path": str(log_path),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
    }


def format_summary(
    args,
    timestamp: str,
    results: list[dict[str, str | None | dict[str, float | int | None]]],
) -> str:
    lines = [
        f"Run Timestamp (Beijing): {timestamp}",
        f"Model: {args.model}",
        f"Requests: {args.num_seqs}",
        f"Input Length: up to {args.max_input_len}",
        f"Output Length: up to {args.max_output_len}",
        "",
    ]
    for result in results:
        lines.append(f"[{result['backend']}]")
        lines.append(f"Status: {result['status']}")
        lines.append(f"Log File: {result['log_path']}")
        lines.append(f"Started At: {result['started_at']}")
        lines.append(f"Finished At: {result['finished_at']}")
        if result["formatted_metrics"] is not None:
            lines.append(str(result["formatted_metrics"]))
        if result["error"] is not None:
            lines.append(f"Error: {result['error']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    backends = resolve_backends(args.backend)
    timestamp = get_run_timestamp()
    backend_logs, summary_log = build_log_paths(args.log_dir, timestamp, backends)

    seed(0)
    prompt_token_ids, max_tokens = build_request_specs(args)

    results = []
    for backend in backends:
        results.append(
            execute_backend_with_logging(
                args,
                backend,
                prompt_token_ids,
                max_tokens,
                backend_logs[backend],
            )
        )

    summary = format_summary(args, timestamp, results)
    summary_log.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Summary log saved to: {summary_log}")

    if any(result["status"] != "success" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
