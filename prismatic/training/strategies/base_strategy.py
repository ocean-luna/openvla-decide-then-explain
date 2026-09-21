"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# Get training type from environment variable for debug output control
train_type = os.environ.get("TRAIN_TYPE", "think_first")


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # Total steps calculation for ETA (global training steps, not per-device)
        total_steps = (self.epochs * len(dataloader) // overwatch.world_size()) if self.max_steps is None else self.max_steps
        vla_training_start_time = time.time()

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=total_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()


                # === Debug Output (Similar to finetune.py) ===
                if overwatch.is_rank_zero() and metrics.global_step % 10 == 0:
                    # Calculate ETA
                    elapsed_time = time.time() - vla_training_start_time
                    steps_completed = metrics.global_step
                    steps_remaining = total_steps - steps_completed
                    if steps_completed > 0:
                        avg_time_per_step = elapsed_time / steps_completed
                        eta_seconds = avg_time_per_step * steps_remaining
                        eta_str = f"{eta_seconds / 3600:.2f}h"
                    else:
                        eta_str = "N/A"

                    print(f"\n[步数 {metrics.global_step}] 训练指标：")
                    print(f"训练损失（loss）: {loss.item():.4f}")
                    print(f"ETA: {eta_str} (已完成 {steps_completed}/{total_steps} 步, 耗时 {elapsed_time / 60:.2f}分钟)")

                    # Print detailed prediction/GT for first sample in batch
                    if len(batch["input_ids"]) > 0:
                        i = 0
                        inp_ids = batch["input_ids"][i].detach().cpu().tolist()
                        inp_txt = self.vlm.llm_backbone.get_tokenizer().decode(inp_ids, skip_special_tokens=False)

                        logits = output.logits[i]
                        labels = batch["labels"][i].to(logits.device)

                        T_in, V = logits.shape
                        T_lab = labels.shape[0]

                        # Align labels to logits length
                        if T_lab < T_in:
                            pad = torch.full((T_in - T_lab,), -100, device=labels.device, dtype=labels.dtype)
                            labels_aligned = torch.cat([pad, labels], dim=0)
                        else:
                            labels_aligned = labels[:T_in]

                        # Shift for causal LM
                        shift_logits = logits[:-1, :].contiguous()
                        shift_labels = labels_aligned[1:].contiguous()

                        sup_mask = shift_labels != -100

                        pred_ids_sup = shift_logits[sup_mask].argmax(dim=-1).detach().cpu().tolist()
                        gt_ids_sup_aligned = shift_labels[sup_mask].detach().cpu().tolist()

                        pred_txt_sup = self.vlm.llm_backbone.get_tokenizer().decode(pred_ids_sup, skip_special_tokens=False)
                        gt_txt_sup_aligned_txt = self.vlm.llm_backbone.get_tokenizer().decode(gt_ids_sup_aligned, skip_special_tokens=False)

                        # Print supervision interval predictions
                        if sup_mask.any():
                            sup_pos = sup_mask.nonzero(as_tuple=True)[0]
                            a, b = int(sup_pos.min().item()), int(sup_pos.max().item())
                            pred_ids_full_supspan = shift_logits[a:b+1].argmax(dim=-1).detach().cpu().tolist()
                            pred_txt_full_supspan = self.vlm.llm_backbone.get_tokenizer().decode(pred_ids_full_supspan, skip_special_tokens=False)
                        else:
                            a = b = -1
                            pred_txt_full_supspan = ""

                        print(f"\n[长度信息] input长度={len(inp_ids)}, logits长度={T_in}, labels长度={T_lab}, "
                              f"左侧补齐(-100)数量={max(T_in - T_lab, 0)}, "
                              f"监督token数量={int(sup_mask.sum().item())}, "
                              f"监督区间(shift坐标)=[{a}, {b}]")

                        print("\n====================== 详细调试输出 ======================")
                        print(f"[步数 {metrics.global_step}] 样本编号={i}")

                        print("\n【输入 INPUT：input_ids 解码】")
                        print(inp_txt)

                        print("\n【GT（严格对齐loss）：labels_aligned[1:] 里非 -100 的监督目标】")
                        print(gt_txt_sup_aligned_txt)

                        print("\n【PRED：只截取监督区间的全量预测解码】")
                        print(pred_txt_full_supspan)

                        # Print GT and Predicted action tokens based on training type
                        action_token_ids_gt = [t for t in gt_ids_sup_aligned if t >= action_tokenizer.action_token_begin_idx]
                        # Get predicted action tokens from the masked positions
                        if sup_mask.any():
                            pred_ids_masked = shift_logits[sup_mask].argmax(dim=-1).detach().cpu().tolist()
                            action_token_ids_pred = [t for t in pred_ids_masked if t >= action_tokenizer.action_token_begin_idx]
                        else:
                            action_token_ids_pred = []

                        # Print action tokens based on training type
                        if train_type == "answer_first":
                            print(f"GT Action Token IDs (answer_first): {gt_ids_sup_aligned[3:10]}")
                            print(f"Pred Action Token IDs (answer_first): {pred_ids_full_supspan[3:10] if len(pred_ids_full_supspan) > 10 else pred_ids_full_supspan}")
                        elif train_type == "think_first":
                            print(f"GT Action Token IDs (think_first): {gt_ids_sup_aligned[-11:-4] if len(gt_ids_sup_aligned) > 11 else gt_ids_sup_aligned}")
                            print(f"Pred Action Token IDs (think_first): {pred_ids_full_supspan[-11:-4] if len(pred_ids_full_supspan) > 11 else pred_ids_full_supspan}")
                        else:
                            print(f"GT Action Token IDs: {action_token_ids_gt[:10]}")
                            print(f"Pred Action Token IDs: {action_token_ids_pred[:10]}")
                        print("==========================================================\n")

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            continuous_actions_pred_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                continuous_actions_pred_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Update progress bar with ETA
                elapsed_time = time.time() - vla_training_start_time
                steps_completed = metrics.global_step
                steps_remaining = total_steps - steps_completed
                if steps_completed > 0:
                    avg_time_per_step = elapsed_time / steps_completed
                    eta_seconds = avg_time_per_step * steps_remaining
                    eta_str = f"ETA: {eta_seconds / 3600:.1f}h"
                else:
                    eta_str = "ETA: --"

                progress.set_description(f"{status} | {eta_str}")

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps) or (epoch >= self.epochs)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
