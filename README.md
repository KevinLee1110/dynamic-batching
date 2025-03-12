# dynamic-batching
The official repo for the paper "[Optimizing LLM Inference Throughput via Memory-aware and SLA-constrained Dynamic Batching](https://arxiv.org/abs/2503.05248)"

The increasing adoption of large language models (LLMs) necessitates inference serving systems that can deliver both high throughput and low latency. Deploying LLMs with hundreds of billions of parameters on memory-constrained GPUs exposes significant limitations in static batching methods. Current inference serving systems often treat batch sizes as fixed hyper-parameters, hindering real-time adaptation to varying system conditions. In this paper, we propose a **dynamic batching** method that continuously monitors memory utilization and adheres to service-level agreements (SLAs) to enable real-time batch size configuration adjustment. The method comprises two core components: a memory-aware batch scheduler that dynamically allocates GPU resources and a latency feedback mechanism that optimizes decoding processes under SLA constraints. The numerical experiments demonstrate **throughput gains of 8% to 28% and capacity improvements of 22%** compared to traditional static batching methods, while maintaining full compatibility with existing inference infrastructure. These results highlight the effectiveness of dynamic batching in balancing computational efficiency and quality-of-service requirements for contemporary LLM deployment scenarios.




## Algorithms

### Algorithm 1: Memory constrained dynamic batching

original code in [`vllm/core/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py#L1219) (static batching)

```python
class Scheduler:
    def _schedule_default(self) -> SchedulerOutputs:
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
		    ...
```

adapted code using dynamic batching

```python
import math

class Scheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        # Add new param for dynamic batching, previous max_num_seqs
        self.prev_max_num_seqs = self.scheduler_config.max_num_seqs
    
    def _schedule_default(self) -> SchedulerOutputs:
        new_max_num_seqs = self.scheduler_config.max_num_seqs
        
        # Dynamic batching
        # params (enable_dynamic_batching, dynamic_batching_memory_factor, batchsize_lower, batch_size_upper) can be added in scheduler_config
        if self.scheduler_config.enable_dynamic_batching:
            new_max_num_seqs = self.prev_max_num_seqs

            # Calculate new max_num_seqs when reqs are arrived
            if self.waiting and self.running:
                # we can use gpu_memory_utilization to change self.block_manager.num_total_gpu_blocks
                used_blocks = self.block_manager.num_total_gpu_blocks - self.block_manager.get_num_free_gpu_blocks()
                blocks_pre_seq = used_blocks / len(self.running)
                # memory_factor can be calculated dynamically or just be a constant (e.g., 0.95)
                memory_factor = self.scheduler_config.dynamic_batching_memory_factor
                new_max_num_seqs = math.floor(self.block_manager.num_total_gpu_blocks * memory_factor / blocks_pre_seq)
                # new_max_num_seqs must be greater than len(self.running)
                new_max_num_seqs = max(new_max_num_seqs, len(self.running))
                new_max_num_seqs = min(max(new_max_num_seqs, self.scheduler_config.batch_size_lower),
                                       self.scheduler_config.batch_size_upper)
                self.prev_max_num_seqs = new_max_num_seqs
        
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=new_max_num_seqs,  # update max_num_seqs
        )
		    ...
```



### Algorithm 2: SLA constrained dynamic batching

The online SLA constrained dynamic batching algorithm can be instantiated in accordance with the pseudo-code presented in Algorithm 2 of our paper.



## Citation

If you find our work helpful, feel free to give us a cite.



## License

This code repository is released under the Apache 2.0 License.
