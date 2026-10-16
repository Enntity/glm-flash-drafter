# Prior art and sources

What this project uses or builds on, with licenses. Links point to the
project pages; check them for the current paper versions.

## Speculative decoding

- **Speculative decoding.** Leviathan, Kalman, Matias, "Fast Inference from
  Transformers via Speculative Decoding" (arXiv:2211.17192); Chen et al.,
  "Accelerating Large Language Model Decoding with Speculative Sampling"
  (arXiv:2302.01318). A cheap drafter proposes tokens, the target verifies
  them in one forward pass.
- **EAGLE / EAGLE-3.** Li et al., arXiv:2401.15077 and arXiv:2503.01840;
  code at https://github.com/SafeAILab/EAGLE (Apache-2.0). Drafters
  conditioned on the target's hidden states, including several intermediate
  layers (EAGLE-3). This project's conditioning on five target layers follows
  the same idea through DFlash2.
- **DFlash.** Z Lab, https://github.com/z-lab/dflash (MIT; the repository
  links the paper). Block drafting: a block of mask tokens predicted in
  parallel by a small non-causal model attending to the target's hidden
  states, so one drafter pass proposes the whole block.
- **DFlash2 architecture.** The model definition used here is
  `speculators.models.dflash2` in vllm-project/speculators (Apache-2.0, with
  Z Lab's MIT notice on code from the DFlash repository) and the matching vLLM
  model file `qwen3_dflash2.py` in https://github.com/vllm-project/vllm
  (Apache-2.0): sliding-window non-causal layers, grouped dynamic
  convolutions, and a candidate selector that re-ranks the unary top-k using
  the previous token. We use the architecture and code only.
- **incoai/GLM-5.3-Flash-DFlash2.** https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2,
  CC BY-NC-ND 4.0. The reference drafter in our measurements. None of its
  weights, configuration, tokenizer files or text are used; the guard refuses
  them.

## Training stack

- **Speculators.** https://github.com/vllm-project/speculators (Apache-2.0),
  revision `824125a`: `DFlash2DraftModel`, the KL / TV / selector losses,
  per-position decay (fixed exponential, D-PACE), anchor-block attention masks
  and accepted-length metrics. Installed as a dependency, not vendored.
- **PyTorch** (BSD-3-Clause): flex attention, `torch.compile`, NCCL/gloo.
- **Transformers** (Apache-2.0): Qwen3 layer config and the chat-template
  renderer.

## Target and serving

- **GLM-5.3-Flash.** https://huggingface.co/zai-org/GLM-5.3-Flash (MIT) and
  the NVIDIA NVFP4 quantization https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4
  (MIT). Teacher and target.
- **Atlas.** https://github.com/Enntity/atlas (AGPL-3.0-only): the serving
  engine; branch `exp/drafter-capture-20261014` adds the hidden-state capture.
- **SparkGLM.** https://github.com/Enntity/sparkglm: GLM-5.3-Flash serving
  recipe for two DGX Sparks with Atlas.
- **RigMark.** https://github.com/alexellis/rigmark (MIT): decode benchmark
  used for some results.

## Low-communication distributed training

- **DiLoCo.** Douillard et al., "DiLoCo: Distributed Low-Communication
  Training of Language Models" (arXiv:2311.08105). Inner AdamW steps per
  worker, infrequent outer updates from averaged parameter deltas with
  Nesterov momentum.
- **OpenDiLoCo.** Jaghouar, Ong, Hagemann, "OpenDiLoCo: An Open-Source
  Framework for Globally Distributed Low-Communication Training"
  (arXiv:2407.07852); https://github.com/PrimeIntellect-ai/OpenDiLoCo
  (Apache-2.0).
- **Streaming DiLoCo.** Douillard et al., "Streaming DiLoCo with overlapping
  communication: Towards a Distributed Free Lunch" (arXiv:2501.18512).
  Fragment-wise, staggered synchronization and lower-precision outer
  gradients.
- **INTELLECT-1.** Prime Intellect, "INTELLECT-1 Technical Report"
  (arXiv:2412.01152); https://github.com/PrimeIntellect-ai/prime. A 10B model
  trained across independently operated sites with DiLoCo-style updates,
  including handling of nodes joining and leaving.
