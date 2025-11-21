import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str                                   # 模型名称
    max_num_batched_tokens: int = 16384          
    max_num_seqs: int = 512                      # 并行序列数目
    max_model_len: int = 4096                    # 
    gpu_memory_utilization: float = 0.9          # gpu占有率
    tensor_parallel_size: int = 1                # 张量并行
    enforce_eager: bool = False                  # 是否启用即时执行
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256                # kvcache的block大小
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
