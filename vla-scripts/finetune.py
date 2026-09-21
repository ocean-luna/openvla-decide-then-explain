"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

export PYTHONPATH="${PWD}:${PYTHONPATH}"
export TRAIN_TYPE=answer_first
export REASONING_DATASET_PATH="data/rlds/libero_reasonings.json"

torchrun --standalone --nnodes 1 --nproc-per-node 6 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir data/rlds \
  --dataset_name libero_lm_90 \
  --run_root_dir runs/answer_first \
  --adapter_tmp_dir adapter-tmp/answer_first \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_latest_checkpoint_only False \
  --save_steps 2000

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
train_type = os.environ.get("TRAIN_TYPE", "think_first")

# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 10000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # Reasoning Parameters
    use_reasoning: bool = True                                     # Whether to use reasoning in training conversation

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    run_id_note: Optional[str] = None                               # Extra note for logging

    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.use_reasoning:
        exp_id += "--reasoning"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        use_reasoning=cfg.use_reasoning,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> TensorBoard
    writer = None
    if distributed_state.is_main_process:
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        # Log hyperparameters
        writer.add_text("config", f"""
        VLA Path: {cfg.vla_path}
        Dataset: {cfg.dataset_name}
        Batch Size: {cfg.batch_size}
        Gradient Accumulation Steps: {cfg.grad_accumulation_steps}
        Learning Rate: {cfg.learning_rate}
        Max Steps: {cfg.max_steps}
        LoRA Rank: {cfg.lora_rank if cfg.use_lora else 'N/A'}
        Use Quantization: {cfg.use_quantization}
        Image Aug: {cfg.image_aug}
        """, 0)

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()
            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            # Push Metrics to TensorBoard (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                writer.add_scalar("train/loss", normalized_loss.item(), gradient_step_idx)
                print(f"\n[步数 {gradient_step_idx}] 训练指标：")
                print(f"训练损失（loss）: {normalized_loss}")

                i = 0  # batch 里第 0 条

                # 1) 输入（模型实际看到的输入序列）
                inp_ids = batch["input_ids"][i].detach().cpu().tolist()
                inp_txt = processor.tokenizer.decode(inp_ids, skip_special_tokens=False)

                # 2) logits / labels
                # 注意：不要用 min(T_logit, T_lab) 去截 logits，否则会把后半段（往往是答案）截掉
                logits = output.logits[i]                  # [T_in, V]
                labels = batch["labels"][i].to(device_id)  # [T_lab]（可能比 T_in 短）

                T_in, V = logits.shape
                T_lab = labels.shape[0]

                # 3) 将 labels 对齐到 logits 的长度
                # 常见情形：labels 少了前缀（比如 img token / 系统prompt），这里左侧补 -100 对齐
                if T_lab < T_in:
                    pad = torch.full(
                        (T_in - T_lab,),
                        -100,
                        device=labels.device,
                        dtype=labels.dtype,
                    )
                    labels_aligned = torch.cat([pad, labels], dim=0)  # 左侧补齐
                else:
                    labels_aligned = labels[:T_in]

                # 4) 按 HF causal LM 的方式 shift：logits[t] 预测 labels[t+1]
                shift_logits = logits[:-1, :].contiguous()         # [T_in-1, V]
                shift_labels = labels_aligned[1:].contiguous()     # [T_in-1]

                sup_mask = shift_labels != -100  # 只有这些位置参与监督/计算loss

                pred_ids_sup = shift_logits[sup_mask].argmax(dim=-1).detach().cpu().tolist()
                gt_ids_sup_aligned = shift_labels[sup_mask].detach().cpu().tolist()

                pred_txt_sup = processor.tokenizer.decode(pred_ids_sup, skip_special_tokens=False)
                gt_txt_sup_aligned_txt = processor.tokenizer.decode(gt_ids_sup_aligned, skip_special_tokens=False)

                # 5) 全量预测（可选：前面可能对应 img token 区域，decode 会“不可读/乱码”，属正常现象）
                pred_ids_full = shift_logits.argmax(dim=-1).detach().cpu().tolist()
                pred_txt_full = processor.tokenizer.decode(pred_ids_full, skip_special_tokens=False)

                # 6) 只打印“监督 span”对应的全量预测（更可读，避免前缀 img token 区域）
                if sup_mask.any():
                    sup_pos = sup_mask.nonzero(as_tuple=True)[0]
                    a, b = int(sup_pos.min().item()), int(sup_pos.max().item())  # shift 坐标系下的起止
                    pred_ids_full_supspan = shift_logits[a:b+1].argmax(dim=-1).detach().cpu().tolist()
                    pred_txt_full_supspan = processor.tokenizer.decode(pred_ids_full_supspan, skip_special_tokens=False)
                else:
                    a = b = -1
                    pred_txt_full_supspan = ""

                # 7) 额外：参考用的 GT（把原始 labels 里 -100 去掉后拼起来；不严格对齐 loss，仅作参考）
                gt_ids_full = batch["labels"][i].detach().cpu().tolist()
                gt_ids_sup = [t for t in gt_ids_full if t != -100]
                gt_txt_sup = processor.tokenizer.decode(gt_ids_sup, skip_special_tokens=False)

                print(
                    f"\n[长度信息]"
                    f" input长度={len(inp_ids)}, logits长度={T_in}, labels长度={T_lab}, "
                    f"左侧补齐(-100)数量={max(T_in - T_lab, 0)}, "
                    f"监督token数量={int(sup_mask.sum().item())}, "
                    f"监督区间(shift坐标)=[{a}, {b}]"
                )

                print("\n====================== 详细调试输出 ======================")
                print(f"[步数 {gradient_step_idx}] 样本编号={i}")

                # 如需看输入可取消注释（可能很长）
                print("\n【输入 INPUT：input_ids 解码】")
                print(inp_txt)

                # 参考用：不严格对齐loss，只是把 -100 去掉后看看大概监督了什么文本
                # print("\n【GT参考：labels 去掉 -100 后的解码（仅参考）】")
                # print(gt_txt_sup)

                print("\n【GT（严格对齐loss）：labels_aligned[1:] 里非 -100 的监督目标】")
                print(gt_txt_sup_aligned_txt)

                # print("\n【PRED（严格对齐loss）：对应监督位置的 argmax 预测】")
                # print(pred_txt_sup)

                # print("\n【PRED（未过滤）：对 logits[:-1] 全部位置 argmax 的解码（可能包含img token区）】")
                # print(pred_txt_full)

                print("\n【PRED：只截取监督区间的全量预测解码】")
                print(pred_txt_full_supspan)
                
                if train_type == "answer_first":
                    print(f"GT Action: {gt_ids_sup_aligned[3:10]} \nPred Action: {pred_ids_full_supspan[3:10]}")
                elif train_type == "think_first":
                    print(f"GT Action: {gt_ids_sup_aligned[-11:-4]} \nPred Action: {pred_ids_full_supspan[-11:-4]}")
                else:
                    print(f"GT Action: {gt_ids_sup_aligned} \nPred Action: {pred_ids_full_supspan}")
                print("==========================================================\n")


                


            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break
    # Close TensorBoard writer
    if distributed_state.is_main_process and writer is not None:
        writer.close()

if __name__ == "__main__":
    finetune()
