"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple

import torch

from vllm import cache_ops
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl

logger = init_logger(__name__)

# key_cache, value_cache, key_scale, value_scale
KVCache = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_heads(parallel_config)
        self.dtype = model_config.dtype

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        # half, int8, int4.
        self.kv_quant_type = cache_config.kv_quant_type
        print("Use CacheEngine kv_quant_type = ", self.kv_quant_type)

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        if self.kv_quant_type == 'int4':
            return (
                self.num_heads,
                self.head_size // 2 // x,
                self.block_size,
                x,
            )

        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    def get_key_scale_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.block_size,
        )

    def get_value_block_shape(self) -> Tuple[int, int, int]:
        if self.kv_quant_type == 'int4':
            return (
                self.num_heads,
                self.head_size // 2,
                self.block_size,
            )

        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )

    def get_value_scale_block_shape(self) -> Tuple[int, int]:
        return (
            self.num_heads,
            self.block_size,
        )

    def allocate_gpu_cache(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        block_type = self.dtype
        if self.kv_quant_type == 'int8' or self.kv_quant_type == 'int4':
            key_scale_block_shape = self.get_key_scale_block_shape()
            value_scale_block_shape = self.get_value_scale_block_shape()
            block_type = torch.int8

        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=block_type,
                device="cuda",
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=block_type,
                device="cuda",
            )
            key_scale = None
            value_scale = None
            if self.kv_quant_type == 'int8' or self.kv_quant_type == 'int4':
                key_scale = torch.empty(
                    size=(self.num_gpu_blocks, *key_scale_block_shape),
                    dtype=self.dtype,
                    device="cuda",
                )
                value_scale = torch.empty(
                    size=(self.num_gpu_blocks, *value_scale_block_shape),
                    dtype=self.dtype,
                    device="cuda",
                )
            gpu_cache.append((key_blocks, value_blocks, key_scale, value_scale))
        return gpu_cache

    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        block_type = self.dtype
        if self.kv_quant_type == 'int8' or self.kv_quant_type == 'int4':
            key_scale_block_shape = self.get_key_scale_block_shape()
            value_scale_block_shape = self.get_value_scale_block_shape()
            block_type = torch.int8
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=block_type,
                pin_memory=pin_memory,
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=block_type,
                pin_memory=pin_memory,
            )
            key_scale = None
            value_scale = None
            if self.kv_quant_type == 'int8' or self.kv_quant_type == 'int4':
                key_scale = torch.empty(
                    size=(self.num_cpu_blocks, *key_scale_block_shape),
                    dtype=self.dtype,
                    pin_memory=pin_memory,
                )
                value_scale = torch.empty(
                    size=(self.num_cpu_blocks, *value_scale_block_shape),
                    dtype=self.dtype,
                    pin_memory=pin_memory,
                )
            cpu_cache.append((key_blocks, value_blocks, key_scale, value_scale))
        return cpu_cache

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache, src_key_scale, src_value_scale = src[i]
                dst_key_cache, dst_value_cache, dst_key_scale, dst_value_scale = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(src_value_cache, dst_value_cache,
                                      src_to_dst)
                if src_key_scale is None:
                    dst_key_scale = None
                else:
                    # Copy the key scale.
                    cache_ops.swap_blocks(src_key_scale, dst_key_scale, src_to_dst)
                if src_value_scale is None:
                    dst_value_scale = None
                else:
                    # Copy the value scale.
                    cache_ops.swap_blocks(src_value_scale, dst_value_scale, src_to_dst)

                event = self.events[i]
                event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _, _, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache, _, _ in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)

        # copy scale id not None
        key_scales = [key_scale for _, _, key_scale, _ in self.gpu_cache]
        value_scales = [value_scale for _, _, _, value_scale in self.gpu_cache]
        if key_scales[0] is not None and value_scales[0] is not None:
            cache_ops.copy_blocks(key_scales, value_scales, src_to_dsts)
        else:
            pass

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        kv_quant_type: str,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        dtype_size = _get_dtype_size(model_config.dtype)
        scale_dtype_size = dtype_size

        key_cache_block = block_size * num_heads * head_size
        if kv_quant_type == 'int4':
            key_cache_block = block_size * num_heads * head_size//2
        value_cache_block = key_cache_block
        key_scale_block = 0
        value_scale_block = 0
        if kv_quant_type == 'int8' or kv_quant_type == 'int4':
            key_scale_block = block_size * num_heads
            value_scale_block = key_scale_block
            dtype_size = _get_dtype_size(torch.int8)

        cache_total = num_layers * (key_cache_block + value_cache_block)
        scale_total = num_layers * (key_scale_block + value_scale_block)

        return dtype_size * cache_total + scale_dtype_size * scale_total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
