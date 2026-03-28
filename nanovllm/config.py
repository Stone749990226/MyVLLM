import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    # 每批次允许的最大 token 数量，用于控制批处理的大小
    # 序列1: [I, love, coding] -> 3 tokens
    # 序列2: [Python, is, great] -> 3 tokens
    # 序列3: [Deep, learning, is, fun] -> 4 tokens
    # 那么这 3 个序列的总 token 数量为 3 + 3 + 4 = 10 tokens，最大不超过 max_num_batched_tokens
    max_num_batched_tokens: int = 16384
    # 每批次允许的最大序列数量
    max_num_seqs: int = 512
    # 模型支持的最大序列长度，超过这个长度的序列会被截断。
    # 假设 max_model_len = 5，有以下序列
    # 序列: [I, love, deep, learning, with, Python]
    # 该序列的长度是 6，超过了 max_model_len，因此会被截断为：
    # [I, love, deep, learning, with]
    max_model_len: int = 4096
    # GPU 内存的利用率上限，用于控制显存分配
    gpu_memory_utilization: float = 0.9
    # 张量并行的大小，用于分布式训练
    tensor_parallel_size: int = 1
    # 是否强制使用 eager 模式（非图模式）
    enforce_eager: bool = False
    # Hugging Face 模型的配置对象，会在初始化时从 model 加载
    hf_config: AutoConfig | None = None
    # 结束标记（End of Sequence）的 token ID
    eos: int = -1
    # KV 缓存的块大小，必须是 256 的倍数
    kvcache_block_size: int = 256
    # KV 缓存的块数量，-1 表示动态分配
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
