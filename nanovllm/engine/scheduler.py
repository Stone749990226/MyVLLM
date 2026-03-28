from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # 先尝试Prefill。self.waiting 里是还没 prefill 的序列
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            # 只把“需要真正算 KV 的 token”算进 batch（前缀命中的部分不再算）
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        # 只要 waiting 里有人且能一起 prefill，就优先搞预填充；一旦有 prefill，就不做 decode
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # Decode时KV Cache不够了，就把队尾的序列抢占回waiting队列，释放它的block。被抢占的序列下次要重新prefill。
            # 如果再给这条 seq 追加 1 个 token，会缺 KV block，那就得“腾位置”
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 从队尾抢占一条，把它 deallocate 掉，放回 waiting
                    self.preempt(self.running.pop())
                else:
                    # 如果已经没人可抢（只剩当前这条），就 preempt(seq) 自己（它也回 waiting），然后 break 掉内层循环，相当于这条暂时 decode 不了
                    self.preempt(seq)
                    break
            else:
                # 只有当 while not can_append 完全没触发 break（也就是最终 can_append(seq) 变成 True）时，才会走 else
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 把这轮参与 decode 的 seq 再插回 running 的左边（队头），保持一个合理的顺序（谁先被取出，又按顺序放回去）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
