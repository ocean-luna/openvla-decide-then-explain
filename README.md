# Decide-then-Explain · OpenVLA

**Where Do Embodied Decisions Come From? Rethinking Latent and Explicit Reasoning**

Yuan Lin, Ziyue Zhou, JinLong Zhao, Pei Liu, Haipeng Liu\*, Pan Zhou, Kun Zhan

Li Auto, Beijing, China · \*Corresponding author

**Decide first, explain afterward: use explicit reasoning as a training signal without making it an intermediate step in real-time action generation.**

This repository contains the code for the paper's **OpenVLA / LIBERO-90 robotic manipulation experiments**, including CoT data integration, training objectives for two generation orders, LoRA fine-tuning, and LIBERO evaluation entry points. It builds on [OpenVLA](https://github.com/openvla/openvla) and [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms), retaining their upstream code structure.

[Repository](https://github.com/ocean-luna/openvla-decide-then-explain) · [Overview](#paper-overview) · [Method](#method-and-configuration) · [Results](#release-scope-and-paper-results) · [Installation](#installation) · [Data and Training](#data-and-training) · [Evaluation](#evaluation) · [Citation](#citation)

> **Release status: research code, not yet a complete reproduction package.** A public paper link, paper checkpoints, and accompanying dataset download links are not yet provided. The local action decoder is not yet aligned with the CoT output format, and the evaluation protocol differs from the paper. We distinguish reported paper results from the capabilities of the released code below. See the [reproduction guide](docs/reproduction.md) and [release checklist](docs/release_checklist.md) (both currently in Chinese).

## Paper Overview

Where should explicit Chain-of-Thought (CoT) appear in an embodied policy? The paper distinguishes **verbalized reasoning** from **perception-grounded internal decision computation**, studying their roles in action generation across autonomous driving and robotic manipulation.

- **CoT benefits depend on model capability and reasoning quality.** CoT can help weaker models but may be neutral or harmful for stronger ones. Improving CoT quality can still yield gains; the conclusion is not that explicit reasoning is always useless.
- **CoT placed before an action influences the decision.** Under think-then-act, replacing the reasoning with a semantically incorrect trace while holding visual inputs and model parameters fixed reduces Navigation F1 from **72.14% to 11.84%** in the driving task. This provides evidence of the strong causal influence of preceding reasoning on downstream decisions.
- **Place explanations after decisions.** Decide-then-explain retains joint supervision over actions and explanations, while preventing the current action from depending on an explanation that has not yet been generated. The paper reports improvements over the corresponding standard think-then-act and no-CoT baselines in both driving and robotics.
- **Characterize dependence on different information sources.** Visual Conditional Contribution (VCC) and Reasoning Conditional Contribution (RCC) measure changes in the log-probability of ground-truth decisions under visual and reasoning-content interventions. They do not directly measure the model's "amount of reasoning," and attention ratios are not causal explanations.

TTA and DTE change the autoregressive factorization during both training and inference. Their comparison is therefore a **structural probe**, not by itself a pure inference-time causal intervention on a fixed policy or proof of a unique "latent reasoning" mechanism. The driving experiments and contribution analyses above provide context for the paper; **their implementations are not included in this repository**.

## Method and Configuration

### Generation Order and Training Objective

Given visual observations $v$, a task instruction $p$, an action $d$, and a reasoning trace $r$, the two strategies use different autoregressive factorizations:

**Think-then-act (TTA):** Generate reasoning first, then predict the action conditioned on that reasoning.

$$
p_\theta(r,d\mid v,p)=p_\theta(r\mid v,p)\,p_\theta(d\mid r,v,p)
$$

**Decide-then-explain (DTE):** Predict the action from observations and the instruction first, then explain it.

$$
p_\theta(d,r\mid v,p)=p_\theta(d\mid v,p)\,p_\theta(r\mid d,v,p)
$$

Both strategies supervise action and reasoning tokens throughout the response. The joint DTE loss is:

$$
\mathcal{L}_{\mathrm{DTE}}=-\log p_\theta(d\mid v,p)-\log p_\theta(r\mid d,v,p)
$$

Causal attention prevents action tokens from attending to the subsequent explanation, while the explanation loss can still provide a training signal through shared parameters. **DTE neither removes CoT supervision nor consists solely of changing the inference prompt.** A post-hoc explanation is not guaranteed to faithfully recover the internal decision process.

| Configuration value | Paper terminology | Training target format |
| --- | --- | --- |
| `think_first` | Think-then-act (TTA) | `<think>Reasoning</think><answer>Action tokens</answer>` |
| `answer_first` | Decide-then-explain (DTE) | `<answer>Action tokens</answer><think>Explanation</think>` |

- `TRAIN_TYPE` controls the training prompt and target sequence format; see [datasets.py](prismatic/vla/datasets/datasets.py).
- `EVAL_TYPE` controls only the evaluation prompt format; see [openvla_utils.py](experiments/robot/openvla_utils.py). It does not change the training strategy or automatically implement output parsing.
- Both default to `think_first` and must be set through environment variables before Python starts. Standard same-strategy evaluation should match the checkpoint's training mode.
- A complete no-CoT data branch is not currently implemented. Neither `use_reasoning=False` nor `TRAIN_TYPE=no_cot` is a working baseline recipe.

### Decision Latency

When only the action is needed, DTE structurally permits generation to stop after `</answer>`, reducing the generation length required to obtain an action from $T_r+T_d$ to $T_d$. If both strategies generate the full action and explanation, the total token count does not decrease. The paper's robotics setting averages 172 reasoning tokens and 7 action tokens, giving an estimated **26× ratio in generation steps**, not an end-to-end speedup measured with this repository.

**The local [predict_action](prismatic/extern/hf/modeling_prismatic.py) still limits generation length to the action dimensionality and decodes the trailing tokens; it does not implement `<answer>` extraction or the corresponding stopping criterion.** Setting `EVAL_TYPE=answer_first` alone does not establish the latency benefit above.

## Release Scope and Paper Results

This repository covers the OpenVLA experiments in *Generalization Across Embodied Domains* and Appendix D.2, *OpenVLA Experiment Details*. It does not include driving data or training pipelines, VCC/RCC metrics, CoT causal interventions, or general-purpose video VLM evaluation.

The following results are taken from the paper's table titled *Generalization to robotic manipulation tasks using OpenVLA on LIBERO-90*. **They were not remeasured using the current repository version.**

| Model | Strategy | LIBERO-90 success rate (%) ↑ |
| --- | --- | ---: |
| OpenVLA baseline | w/o CoT | 62.0 |
| OpenVLA + CoT | Think-then-act | 67.1 |
| OpenVLA + CoT | **Decide-then-explain** | **69.2** |

DTE improves over TTA by **2.1 percentage points** and over the no-CoT baseline by **7.2 percentage points**. The paper reports a standard deviation below 0.5 percentage points for DTE success rate. The baseline row cites Belkhale and Sadigh's [MiniVLA / openvla-mini](https://github.com/Stanford-ILIAD/openvla-mini). This repository does not provide a directly runnable no-CoT baseline recipe or the complete experiment artifacts underlying these results.

The paper evaluates 90 tasks with 10 trials per task and at most 400 action steps per episode, excluding anomalous episodes in which every action is a zero vector. **The current evaluation code does not implement this episode-level exclusion rule.** Reproduction reports must disclose the original episode count, exclusions, and evaluation denominator, and retain unfiltered results. Existing script outputs should not be treated as directly equivalent to the table above.

<a id="安装"></a>

## Installation

The target environment is **Linux + NVIDIA CUDA + Python 3.10**. Core versions are PyTorch 2.2.0, torchvision 0.17.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, PEFT 0.11.1, and TensorFlow 2.15.0. See [pyproject.toml](pyproject.toml) for the complete dependencies. CUDA training and simulation have not been validated as part of this repository cleanup; macOS training support is not guaranteed.

Clone the repository, then run the installation commands from its root:

```bash
git clone https://github.com/ocean-luna/openvla-decide-then-explain.git
cd openvla-decide-then-explain

conda create -n openvla-dte python=3.10 -y
conda activate openvla-dte

# Example CUDA 12.1 wheels; ensure compatibility with your driver and local CUDA toolkit.
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install packaging ninja
pip install flash-attn==2.5.5 --no-build-isolation
```

The distribution name is `openvla-decide-then-explain`, while Python imports still use `prismatic`. Do not share this environment with another OpenVLA installation that provides the same Python package.

Install LIBERO and configure its resource paths following the [official repository](https://github.com/Lifelong-Robot-Learning/LIBERO), then run:

```bash
pip install -r experiments/robot/libero/libero_requirements.txt
```

The evaluation code sets `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`, requiring Linux OSMesa runtime libraries. Setting `egl` externally is not a supported backend switch in the current code; see the [reproduction guide](docs/reproduction.md).

## Data and Training

### Data Preparation

The paper uses the LIBERO-90 reasoning dataset adopted by ECoT-Lite, containing **90 tasks and 3,917 successful demonstration trajectories**. Each step includes a third-person RGB observation, a natural-language task instruction, a 7-dimensional action, and structured CoT annotations such as plans, subtasks, movement reasoning, object bounding boxes, and gripper position.

This repository does not distribute these data or provide verified download links for the paper's datasets or weights. Before training, prepare:

1. **Compatible RLDS / TFDS data.** The launcher's default registered dataset name is `libero_lm_90`, and `DATA_ROOT` points to the data root. This name is not a public download identifier; renaming an original LIBERO directory does not produce a compatible dataset.
2. **Per-step reasoning JSON.** Set `REASONING_DATASET_PATH` and align annotations with each trajectory's `episode_metadata.file_path`, `demo_id`, and zero-based step index. Missing matches yield empty reasoning text, so check annotation coverage before training.

See the [data interface](docs/reproduction.md#数据接口) for field definitions, directory layout, and a JSON example. The [demonstration regeneration script](experiments/robot/libero/regenerate_libero_dataset.py) replays demonstrations and outputs HDF5; **it does not include HDF5-to-RLDS conversion or CoT annotation generation**. Per-step no-op filtering during data preprocessing is distinct from episode-level exclusion during the paper's evaluation.

### Paper Training Configuration

The following configuration comes from Appendix D.2 and applies only to the robotics experiments, not to the driving models:

| Component | Paper setting |
| --- | --- |
| Backbone | OpenVLA: Llama2-7B + dual DINOv2 / SigLIP vision encoders |
| Action space | 7 dimensions: translation deltas, rotation deltas, and gripper state, discretized with the action tokenizer |
| Fine-tuning | LoRA, rank = 32 |
| Hardware and duration | 32 NVIDIA H20 GPUs, approximately 78 hours |
| Optimization steps | 200,000 |
| Per-GPU batch size / gradient accumulation | 6 / 1, effective batch size = 192 |
| Optimizer and learning rate | AdamW, 8e-4, no warmup |

The paper does not specify the image augmentation flag. `IMAGE_AUG=False` below follows the launcher default and should not be taken as confirmation of the paper's setting.

### LoRA Fine-Tuning

The following is a single-node, single-GPU configuration example, not the paper's 32-GPU setup or an end-to-end validated run. Paths are relative to the repository root; adjust them to your data locations:

```bash
export DATA_ROOT=data/rlds
export DATASET_NAME=libero_lm_90
export REASONING_DATASET_PATH="${DATA_ROOT}/libero_reasonings.json"
export VLA_PATH=openvla/openvla-7b
export NPROC_PER_NODE=1
export NODE_NUM=1
export NODE_RANK=0
export TRAIN_TYPE=answer_first
export RUN_ID_NOTE=answer_first
export BATCH_SIZE=6
export GRAD_ACCUM_STEPS=1
export LORA_RANK=32
export LEARNING_RATE=8e-4
export MAX_STEPS=200000
export SAVE_STEPS=10000
export IMAGE_AUG=False
bash vla-scripts/finetune.sh
```

To train think-then-act, set both `TRAIN_TYPE` and `RUN_ID_NOTE` to `think_first`, keeping other comparison conditions identical. Changing `EVAL_TYPE` alone does not switch the training strategy. GPU memory requirements have not been validated for the current version. Reducing device count or batch size changes the effective batch size and is not a strict reproduction.

By default, training artifacts and TensorBoard logs are saved under `runs/`, with temporary adapters under `adapter-tmp/`. The example explicitly sets the save interval to 10,000 so it divides the total training steps. The default interval of 30,000 does not guarantee a final checkpoint at step 200,000.

A 4-node × 8-GPU setup can match the paper's reported 32-GPU scale; the paper does not specify the node topology. See the [training configuration](docs/reproduction.md#训练配置) for distributed examples, output directories, and credential requirements. The retained `vla-scripts/launch_train.sh` is a Prismatic full fine-tuning entry point, **not the paper's main LoRA recipe**.

## Evaluation

**Address the [reproduction limitations](docs/reproduction.md#复现限制) before using the following entry point.** Before evaluation, confirm that:

- `MODEL_PATH` points to a complete model checkpoint for the intended strategy, not a LoRA-adapter-only directory. The original `openvla/openvla-7b` is not a substitute for the paper's fine-tuned model.
- The action decoder actually loaded has been verified to handle `<think>` / `<answer>` outputs correctly. The loader uses `trust_remote_code=True`, so checkpoints may contain model code different from the local implementation. Load only trusted resources.
- The checkpoint's action statistics match the training data. Training uses the registered name `libero_lm_90`, while evaluation looks for `libero_90` or `libero_90_no_noops`. There is no automatic mapping; do not blindly rename statistics keys.
- You have accounted for the episode-level zero-action filtering difference described above and checked the image preprocessing required by the checkpoint.

```bash
export MODEL_PATH=checkpoints/answer-first-checkpoint
export EVAL_TYPE=answer_first
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TASK_SUITE=libero_90
export TOTAL_TASKS=90
export GPU_COUNT=1
export NUM_TRIALS_PER_TASK=10
export SEED=7
export CENTER_CROP=False
export LOG_DIR=logs/answer_first_seed7
export ROLLOUT_DIR=rollouts/answer_first_seed7
bash vla-scripts/launch_libero_eval.sh
```

`CENTER_CROP=False` corresponds to the `IMAGE_AUG=False` training example above. For checkpoints trained with augmentation, configure cropping according to their actual preprocessing. Evaluation defaults to 10 trials per task, at most 400 action steps per episode, and 10 additional settling steps. The GPU keep-alive feature is disabled by default and is unnecessary for normal evaluation.

For TTA evaluation, use the corresponding TTA checkpoint, set `EVAL_TYPE=think_first`, and use separate log and video directories. Increasing `GPU_COUNT` distributes tasks across independent processes on GPUs `0..GPU_COUNT-1`; ensure those devices are allocated to your job. The script does not aggregate metrics automatically. Compute success rate as **total successes / total trials**, checking complete task coverage and error logs rather than averaging per-GPU percentages or relying only on the launcher's completion message.

## Repository Structure

```text
prismatic/                        Upstream framework and project data, training, and model code
vla-scripts/finetune.py            LoRA fine-tuning entry point
vla-scripts/finetune.sh            Configurable LoRA launcher
vla-scripts/launch_train.sh        Retained Prismatic full fine-tuning launcher
vla-scripts/launch_libero_eval.sh  Multi-GPU LIBERO task distribution
experiments/robot/                Model loading and robot environment evaluation
scripts/                         Retained upstream general-purpose VLM tools
docs/reproduction.md              Data, configuration, and reproduction limitations (Chinese)
docs/release_checklist.md          Pre-release security and validation checklist (Chinese)
```

## Citation

If this project is useful for your research, please cite our paper. The following BibTeX entry reflects the title and authors in the current manuscript; publication details and a paper link will be added once confirmed:

```bibtex
@misc{lin_decide_then_explain,
  title={Where Do Embodied Decisions Come From? Rethinking Latent and Explicit Reasoning},
  author={Yuan Lin and Ziyue Zhou and JinLong Zhao and Pei Liu and Haipeng Liu and Pan Zhou and Kun Zhan},
  note={Research manuscript}
}
```

## License and Acknowledgments

The code retains the [MIT License](LICENSE) and the original OpenVLA copyright notice. MIT does not automatically grant redistribution rights for model weights or datasets. The Llama-2 model underlying OpenVLA and the individual data sources have their own licenses and access conditions. Access to the original Llama-2 model ID may require separate authorization.

We thank OpenVLA, Prismatic VLMs, LIBERO, and ECoT-Lite for their open-source work. Please also cite the upstream projects your work relies on. The OpenVLA citation is:

```bibtex
@article{kim24openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}
```
