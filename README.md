# parallelism-from-scratch

[![ci](https://github.com/GonzaloFuentes1/parallelism-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/GonzaloFuentes1/parallelism-from-scratch/actions/workflows/ci.yml)

Readable implementations of the parallelism strategies used in LLM pretraining (data
parallelism with bucketed all-reduce, ZeRO stages 1 and 2, Megatron-style tensor
parallelism with a vocab-parallel loss, GPipe and 1F1B pipeline schedules, and sharded
checkpoints with resharding), applied to a tiny Llama-style model. Every strategy is
checked against a single-process float64 reference by parity tests that run on CPU,
with `torch.distributed` over gloo and one process per rank, so the whole suite runs
on a laptop or a free CI runner.

## What it is, and what it is not

It is a reference for *what* each strategy communicates and *when*, written so that
the code reads like the explanation, with tests that make the equivalence to
single-process training a checked fact rather than a claim.

It is not a training framework. There are no fused kernels, no mixed precision, no
communication/computation stream management beyond async collectives, and no
performance work. For real training use
[torchtitan](https://github.com/pytorch/torchtitan),
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM),
[nanotron](https://github.com/huggingface/nanotron) or
[DeepSpeed](https://github.com/deepspeedai/DeepSpeed).

## Strategies at a glance

`P` = parameter count, `N` = ranks, `K = 12` bytes/param of fp32 optimizer state
(master weight, momentum, variance) under mixed-precision Adam. Memory is model state
per rank (parameters + gradients + optimizer), from `cost_model.py`; communication is
the payload per training step under the convention in `comm.py`.

| Strategy | Module | What is sharded | Communication per step | Model-state memory per rank |
|---|---|---|---|---|
| DDP | `ddp.py` | nothing | all-reduce of all gradients (`P`) | `(2 + 2 + K) P = 16 P` |
| ZeRO-1 | `zero.py` | optimizer state | reduce-scatter of gradients + all-gather of parameters (same ring volume as DDP) | `4 P + K P / N` |
| ZeRO-2 | `zero.py` | optimizer state, gradients | same as ZeRO-1 | `2 P + (2 + K) P / N` |
| ZeRO-3 | cost model only | also parameters | as ZeRO-1 plus a second parameter all-gather (1.5x DDP) | `(2 + 2 + K) P / N` |
| Tensor parallel | `tensor_parallel.py` | every weight matrix, the vocabulary | 2 activation all-reduces per layer in forward, 2 in backward | about `16 P / N` (norm weights replicated) |
| Pipeline | `pipeline.py` | layers | point-to-point activations forward, their gradients backward, per microbatch and stage boundary | about `16 P / N` |

ZeRO-3 appears only in the cost model; it is not implemented here. The memory column
counts the buffers kept between steps; `test_cost_model.py` checks its float64
analogue against the buffers DDP and ZeRO-1/2 actually hold after a step.

## Design notes

**Reference model** (`model.py`). Pre-norm Llama: RMSNorm, rotate-half RoPE, SwiGLU,
causal multi-head attention written out explicitly, untied output head. It runs in
float64 so parity can be asserted at `1e-12`. `Attention` and `MLP` read their head
count and projections from attributes; the tensor-parallel subclasses only replace
those, so the forward code is shared and cannot drift from the reference.

**DDP** (`ddp.py`). Rank 0's weights are broadcast at construction. Parameters are
grouped into buckets in reverse registration order, the order in which backward
produces their gradients. A post-accumulate-grad hook launches an asynchronous
all-reduce as soon as a bucket is complete, so communication for late layers overlaps
with backward of early layers. Each parameter's `.grad` is a view into its bucket
buffer (as with `gradient_as_bucket_view` in PyTorch), so the all-reduce runs in place
and there is no second gradient copy; `finish_gradient_sync()` waits and divides the
buffers by `N`. Averaging gradients of equal-sized local batches equals the
gradient of the mean loss over the concatenated batch.

**ZeRO-1 and ZeRO-2** (`zero.py`, `optim.py`). Parameters are grouped into buckets in
the same backward order as DDP; each bucket is a flat buffer padded to a multiple of `N`
and cut into `N` contiguous shards, and parameters become views into these buffers. A
reduce-scatter per bucket leaves each rank with the averaged gradient of its shards;
AdamW updates only those shards (legal because AdamW is elementwise, so shard
boundaries need not align with parameters); an all-gather per bucket writes the updated
shards back into every rank's parameters. Stage 1 keeps the full `.grad` of every
parameter and reduce-scatters inside `step()`, one bucket-sized copy at a time. Stage 2
reduce-scatters each bucket from a post-accumulate-grad hook during backward and
releases that bucket's full gradients at once, accumulating the averaged shards into a
persistent buffer of `P / N`. During backward the full gradients alive are the buckets
still being filled plus one bucket in flight; when backward returns no parameter holds
a `.grad`.

**Tensor parallelism** (`tensor_parallel.py`). Column-parallel `wq`, `wk`, `wv`, `w1`,
`w3` and `lm_head` split output features; for attention that is whole heads, so
attention runs locally. Row-parallel `wo` and `w2` split input features and sum
partial outputs with one all-reduce. The matching backward all-reduce sits on the norm
that feeds each parallel region, so `q`, `k`, `v` share one instead of three. The
embedding is split by vocabulary rows (out-of-range tokens contribute zero, one
all-reduce sums). The loss never forms full logits: the vocab-parallel cross-entropy
all-reduces the row max (for stability), the target logit and the softmax normalizer,
and its backward is the local slice of `softmax - one_hot`.

**Distributed checkpointing** (`checkpoint.py`). Each local shard is wrapped as a
`DTensor` whose placement comes from the same `shard_dim` table used to load weights,
and `torch.distributed.checkpoint` writes every rank's shards in parallel with metadata
describing global shapes. Loading at another tensor-parallel size reads only the
slices each rank needs. Consolidation is the single-process case of the same mechanism:
full-size tensors built from the checkpoint metadata are loaded with `no_dist=True`
and written to one safetensors file.

**Pipeline parallelism** (`pipeline.py`). Stages are contiguous layer ranges; the
first owns the embedding, the last the norm, head and loss. Schedules are explicit
action lists executed by one function: GPipe runs all forwards then all backwards;
1F1B warms up with `S - s - 1` forwards and then alternates, freeing each
microbatch's activations after its backward, so stage `s` holds at most `S - s`
microbatches instead of `M`. Sends are asynchronous because in the 1F1B steady state
neighbouring stages both send before they receive.

**Cost model** (`cost_model.py`). The ZeRO memory formulas with `ceil(P / N)`
padding, collective payloads per step, ring bytes per rank, and tensor-parallel
parameter count and activation all-reduce payload.

## How it is verified

All distributed tests spawn real processes (`launch.run_distributed`, gloo over a
per-test `FileStore`) and assert inside each rank. Tolerances are absolute, at float64.

| Test file | What it checks |
|---|---|
| `test_model.py` | attention equals `F.scaled_dot_product_attention`; RMSNorm equals `F.rms_norm`; RoPE equals the Hugging Face rotate-half formula bitwise and its scores depend only on relative position; logits are causal; initial loss is close to `log(V)`; seeded init is deterministic; AdamW lowers the loss |
| `test_comm.py` | all-reduce, reduce-scatter, all-gather and max match closed-form results at 2 and 4 ranks; metered payloads; a failure on one rank fails the caller |
| `test_ddp.py` | gradients at every step and parameters after 3 AdamW steps equal one process on the concatenated batch (`atol=1e-12`) at 2 ranks with several buckets and 4 ranks with one; non-zero ranks start from other weights to prove the broadcast; one all-reduce per bucket covering the model once; every all-reduce is launched before backward returns, and with several buckets at least one is in flight while gradients are still pending; every `.grad` shares storage with its bucket buffer |
| `test_zero.py` | functional AdamW equals `torch.optim.AdamW`; AdamW on disjoint slices equals AdamW on the whole (bitwise); with several buckets, ZeRO-1 at 2 ranks and ZeRO-2 at 3 ranks (per-bucket padding) equal unsharded AdamW after every step; both gathered moment shards equal the reference moments; ZeRO-2's gathered gradient shards equal the full-batch gradient; ZeRO-2 reduce-scatters every bucket before backward returns, leaves no `.grad`, and at every gradient hook holds no full gradient of an already reduced bucket and at most twice the largest bucket in total; payload is one reduce-scatter and one all-gather per bucket covering the padded buffer once |
| `test_tensor_parallel.py` | at TP=2 and TP=4: gathered logits, loss, every gradient shard and parameters after 3 AdamW steps equal the reference; all-reduce counts in forward and backward match the design; vocab-parallel CE with logits offset by `1e4` is finite and equals `F.cross_entropy` loss and gradients; shards reassemble bitwise |
| `test_checkpoint.py` | save at TP=2, reload at TP=4, re-save, reload at TP=2: shards bitwise equal and logits equal; both checkpoints consolidate to safetensors that are bitwise equal to the saved model with bitwise-equal logits |
| `test_cost_model.py` | the 16 bytes/param baseline; ZeRO paper Figure 1 (7.5B parameters, 64 GPUs: 120, 31.4, 16.6 and 1.9 GB); ZeRO-1 ring volume equals DDP; formulas equal the storage of the buffers DDP and ZeRO-1/2 keep after a step at 3 ranks (counted by storage, so DDP's bucket buffers and ZeRO's padding are included; transient bucket copies are not) and the metered payloads; TP parameter count and all-reduce payload equal what is allocated and sent |
| `test_pipeline.py` | 1F1B is a valid FIFO reordering of GPipe; both schedules give reference gradients and loss with 2 stages and 4 microbatches and with 4 stages and 8 microbatches (middle stages, 1F1B steady state); peak in-flight microbatches is `M` for GPipe and `S - s` for 1F1B; a batch that does not split into equal microbatches raises `ValueError` |

## Measured results

Largest absolute deviation from the single-process float64 reference, measured with
`python examples/parity_report.py` (tiny Llama with 4 layers, 3 AdamW steps,
`lr=1e-2`) on Windows 11, Intel Core i7-13620H, Python 3.12, torch 2.14.1+cpu:

| Strategy | Ranks | Max abs gradient difference (step 1) | Max abs parameter difference (after 3 steps) |
|---|---|---|---|
| DDP | 2 | 3.5e-17 | 1.6e-14 |
| ZeRO-1 | 2 | n/a | 1.6e-14 |
| ZeRO-2 | 4 | n/a | 8.4e-14 |
| Tensor parallel | 2 | 5.6e-17 | 2.8e-14 |
| Tensor parallel | 4 | 5.6e-17 | 2.8e-14 |
| Pipeline, GPipe | 2 | 2.4e-17 | n/a |
| Pipeline, 1F1B | 4 | 2.4e-17 | n/a |

ZeRO never materializes the averaged full gradient, so only its parameters are
compared. The pipeline report covers one forward/backward pass.

The full suite (36 tests) ran in 61.7 s on Python 3.11 and 68.1 s on Python 3.12 on
GitHub Actions `ubuntu-latest` with torch 2.14.1+cpu (`python -m pytest -v
--durations=10`, CI run 39506354157).

## Quickstart

```bash
git clone https://github.com/GonzaloFuentes1/parallelism-from-scratch
cd parallelism-from-scratch
uv venv --python 3.12 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -e ".[dev]"
.venv/bin/python examples/parity_report.py
```

A data-parallel training step, run on 2 CPU processes:

```python
import torch

from parallelism_from_scratch.ddp import DistributedDataParallel
from parallelism_from_scratch.launch import run_distributed
from parallelism_from_scratch.model import LlamaConfig, build_reference, lm_loss, make_batch


def train(rank: int, world_size: int) -> None:
    cfg = LlamaConfig()
    model = DistributedDataParallel(build_reference(cfg), bucket_cap_bytes=4096)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for step in range(3):
        tokens, targets = make_batch(cfg, batch_size=2, seq_len=16, seed=100 * step + rank)
        opt.zero_grad()
        loss = lm_loss(model(tokens), targets)
        loss.backward()
        model.finish_gradient_sync()
        opt.step()
        if rank == 0:
            print(step, loss.item())


if __name__ == "__main__":
    run_distributed(train, 2)
```

## Running the tests

```bash
.venv/bin/python -m pytest            # full suite
.venv/bin/python -m pytest tests/test_tensor_parallel.py -v
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Linux CI is the source of truth. Spawning processes is much slower on Windows, and
gloo on Windows can occasionally fail to connect; each test uses its own `FileStore`
in a temporary directory, so tests never compete for a port.

## Layout

```
src/parallelism_from_scratch/
  model.py            tiny Llama reference
  launch.py           spawn N gloo ranks over a FileStore
  comm.py             metered collectives and point-to-point ops
  ddp.py              bucketed, overlapped gradient all-reduce
  optim.py            functional AdamW on flat tensors
  zero.py             ZeRO stages 1 and 2
  tensor_parallel.py  column/row-parallel layers, vocab-parallel embedding and loss
  checkpoint.py       DCP save/load with resharding, safetensors consolidation
  pipeline.py         pipeline stages, GPipe and 1F1B schedules
  cost_model.py       per-rank memory and communication formulas
examples/parity_report.py
tests/
```

## References

- S. Rajbhandari, J. Rasley, O. Ruwase, Y. He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, B. Catanzaro. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- Y. Huang et al. *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.* [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
- A. Harlap et al. *PipeDream: Fast and Efficient Pipeline Parallel DNN Training* (1F1B). [arXiv:1806.03377](https://arxiv.org/abs/1806.03377)
- D. Narayanan, A. Phanishayee, K. Shi, X. Chen, M. Zaharia. *Memory-Efficient Pipeline-Parallel DNN Training* (PipeDream-Flush). [arXiv:2006.09503](https://arxiv.org/abs/2006.09503)
- PyTorch. *Distributed Checkpoint (torch.distributed.checkpoint).* [docs](https://pytorch.org/docs/stable/distributed.checkpoint.html)

## License

MIT. See [LICENSE](LICENSE).
