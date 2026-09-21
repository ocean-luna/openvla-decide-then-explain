# OpenVLA / LIBERO-90 复现说明

本文件区分论文配置、当前代码行为与尚未验证的环节。配方来源为匿名论文 §4.3.5、Table 6 和附录 D.2；本次整理没有重新训练模型，也没有运行 LIBERO 仿真。

## 复现限制

以下是当前版本的已知限制，不应通过更换 README 的表述掩盖，也未在本次文档与配置整理中修复。

| 项目 | 当前代码证据 | 影响与后续要求 |
| --- | --- | --- |
| 带 CoT 的动作解码 | [本地 HF 模型的 `predict_action`](../prismatic/extern/hf/modeling_prismatic.py) 使用 `max_new_tokens=get_action_dim(...)`，并直接取末尾动作维数个 token | 无法据此正确处理完整 `<think>` / `<answer>` 输出。需要核实实验实际使用的解码器，并公开动作提取与停止条件；当前评估命令仅说明入口配置，不保证有效论文评估。 |
| checkpoint 自定义代码 | [加载入口](../experiments/robot/openvla_utils.py) 使用 AutoClasses 和 `trust_remote_code=True` | 实际 checkpoint 可能携带其他模型实现。需要记录配置、代码版本和实际加载类，不能假定与本地实现一致；只加载可信来源。 |
| 整回合零动作过滤 | [LIBERO 评估循环](../experiments/robot/libero/run_libero_eval.py) 对每个已运行回合累计分母，未实现论文 D.2.4 所述的整回合零动作排除 | 当前统计口径与论文描述不一致。复现时须公开原始回合数、排除数及规则，同时报告未过滤结果；本次不新增过滤或修改结果。 |
| 动作反归一化键 | 训练注册名为 `libero_lm_90`，评估按 `libero_90` 或其 `_no_noops` 后缀查找统计量 | 若 checkpoint 只有 `libero_lm_90` 键，评估将触发断言。必须核实数据和统计量的对应关系，不能盲目重命名键或使用其他数据集的统计值。 |
| no-CoT 对照 | [数据变换](../prismatic/vla/datasets/datasets.py) 中 `use_reasoning` 未控制完整的无推理分支，其他 `TRAIN_TYPE` 值也不是可用的无 CoT 配方 | `use_reasoning=False` 不能视为标准 no-CoT 实验。本文档只给出两个已有顺序的配置入口。 |
| 数据与模型资源 | 未提供已核验的论文 checkpoint、reasoning JSON 下载地址、RLDS 构建版本和转换流程 | 数据 schema 说明不能替代完整数据发布；尚不能承诺从原始 LIBERO 到论文结果的一键复现。 |
| 跨域实验 | 当前分支不包含驾驶实验、VCC/RCC、CoT 干预和视频 VLM 评估模块 | 本仓库仅覆盖 OpenVLA 机器人部分的整理范围。 |

附录 F 的提前截断和约 26 倍机器人决策加速是基于 token 数的分析，不是本仓库已实现、已测得的端到端性能。answer-first 的结构允许先取得动作，但本地解码器尚未完成相应支持。

## 环境与依赖

- 目标环境：Linux、NVIDIA GPU、Python 3.10；推荐独立虚拟环境，避免同时安装多个提供 `prismatic` 包的项目。
- 核心依赖：PyTorch 2.2.0、torchvision 0.17.0、transformers 4.40.1、tokenizers 0.19.1、timm 0.9.10、PEFT 0.11.1、TensorFlow 2.15.0、tensorflow-datasets 4.9.3。以 [pyproject.toml](../pyproject.toml) 为准。
- Flash Attention 2.5.5 单独安装。PyTorch wheel、GPU 驱动和编译 CUDA 工具链需要匹配，不使用仓库内的平台专用 wheel 作为通用安装方式。
- TensorBoard 已加入依赖。LoRA 训练使用本地 TensorBoard，不需要 W&B 项目；评估默认不上传 W&B。
- `dlimp` 的 Git 依赖目前未锁定 commit；正式复现需记录实际安装 commit、LIBERO commit 和完整环境版本。本次不替换未知的实验依赖版本。
- 安装方法见 [README](../README.md#安装)。macOS 静态检查不等同于 Linux CUDA 验证。

当前 [run_libero_eval.py](../experiments/robot/libero/run_libero_eval.py) 在导入 LIBERO 之后设置 OSMesa 环境变量，具体初始化行为还需在目标环境验证。运行前应显式设置：

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

还需安装目标 Linux 发行版提供的 OSMesa 运行库，并完成 LIBERO 的 BDDL、初始化状态等资源配置。当前代码会写入 OSMesa 设置，因此不能声称通过外部 `egl` 环境变量即可切换为 EGL。本次未调整导入顺序或渲染实现。

## 数据接口

### 来源与目录

论文描述使用 ECoT-Lite 的 LIBERO-90 reasoning 数据：90 个任务、3,917 条成功示范轨迹。该描述来自论文，不是对当前本地数据的统计。数据获取、再分发权限、具体版本以及转换程序均需单独确认。

训练启动脚本默认 `DATASET_NAME=libero_lm_90`。这是本仓库的 TFDS 注册名称，不是公开下载标识；也不能靠重命名任意目录将其他数据集转换为该数据集。

```text
data/rlds/
├── libero_lm_90/
│   └── <实际 TFDS 版本>/
│       ├── dataset_info.json
│       └── <TFRecord 及其他 TFDS 文件>
└── libero_reasonings.json
```

- `DATA_ROOT` 必须指向 TFDS 数据根目录，由用户显式设置。
- 启动脚本默认 `REASONING_DATASET_PATH=${DATA_ROOT}/libero_reasonings.json`，可覆盖为单独存放的标注文件。
- 直接运行 Python 时，默认路径为 `data/libero_reasonings.json`；该默认值以及生成顺序在模块导入时读取环境变量，必须在启动 Python 之前设置。
- 当前数据加载器会读取 reasoning JSON，不提供缺失时自动下载或退化为无推理训练的保证。

### RLDS 字段与匹配

接口依据：[数据加载器](../prismatic/vla/datasets/rlds/dataset.py)、[LIBERO 注册配置](../prismatic/vla/datasets/rlds/oxe/configs.py)、[标准化变换](../prismatic/vla/datasets/rlds/oxe/transforms.py)。

原始轨迹需要包含动作、语言指令和符合注册配置的 observation。其中主视角映射自 `image`，LIBERO 标准化代码还读取 `state`。经标准化后，batch 变换读取 `observation.image_primary`、`task.language_instruction`、`task.reasoning` 和 `action`。动作是 7 维：平移增量、旋转增量和夹爪状态；实际归一化与夹爪约定沿用当前代码，不在此重新定义。

每条轨迹还需携带 `episode_metadata.file_path` 和 `episode_metadata.demo_id`；加载器兼容外层的 `traj_metadata.episode_metadata`。匹配使用：

```text
文件 basename + "_" + demo_id 的字符串 + "_" + 轨迹内从 0 开始的 step index
```

因此 JSON 顶层必须与 `file_path` 的文件名一致，episode 键必须与 `demo_id` 的字符串表示一致，step 键必须匹配 RLDS 的步序。重采样、删除 no-op 或轨迹裁剪后，需要重新核验对齐，不能沿用不对应的标注索引。

### Reasoning JSON 示例

下面仅演示结构，不是真实样本，也不是可用于训练的数据集：

```json
{
  "example_task_demo.hdf5": {
    "0": {
      "0": {
        "plan": {"0": "接近目标", "1": "抓取目标"},
        "subtask": "接近目标",
        "subtask_reasoning": "当前尚未到达抓取位置",
        "movement": "向目标移动",
        "movement_reasoning": "根据当前观测缩短与目标的距离",
        "bboxes": {"target": [[10, 20], [30, 40]]},
        "gripper": [20, 30]
      }
    }
  }
}
```

当前实现将这些字段序列化为 `PLAN@...@SUBTASK@...@SUBTASK_REASONING@...` 等文本，随后在 batch 变换中转为小写。它不是直接拼接上述 JSON。查找未命中的 step 会得到空字符串，因此正式训练前应统计匹配覆盖率；本次不修改匹配或增加数据筛选。

## 训练配置

### 论文配置与当前默认值

| 项目 | 论文附录 D.2 | 当前 LoRA 启动脚本默认值 |
| --- | --- | --- |
| 基座 | OpenVLA：Llama2-7B + DINOv2 / SigLIP | `openvla/openvla-7b`，不是论文微调权重 |
| 硬件 | 32 张 NVIDIA H20，约 78 小时 | 单节点，8 进程；硬件与耗时未经本次验证 |
| 每卡 batch | 6 | 6 |
| 梯度累积 | 1 | 1 |
| 有效 batch | 192 | 48；取决于实际 world size |
| LoRA rank | 32 | 32 |
| 学习率 | 8e-4 | 8e-4 |
| 优化步数 | 200,000 | 200,000 |
| 优化器/调度 | AdamW，无 warmup | AdamW，固定学习率 |
| 图像增强 | D.2 未明确给出此开关 | `IMAGE_AUG=False`；Python 类默认值则为 True |
| 保存间隔 | D.2 未指定 | 30,000；下面示例显式设为 10,000 |
| 生成顺序 | 两种策略分别训练 | `TRAIN_TYPE=think_first` |

有效 batch = 每卡 batch × 梯度累积 × 节点数 × 每节点进程数。不要将学习率相同但有效 batch 不同的运行称为严格复现。

### 单机入口

以下配置用于验证入口，不代表本次已验证它可完成训练，也不等同于论文算力配置：

```bash
export DATA_ROOT=data/rlds
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

### 论文规模：4 节点 × 8 卡

在每个节点的仓库根目录设置下面相同配置；`MASTER_ADDR` 替换为各节点均可访问的主节点地址，`NODE_RANK` 在四台节点分别设为 0、1、2、3。所有节点须能访问一致的数据和模型，并为现有 checkpoint/adapter 保存流程提供共享输出目录。示例中的 `data/rlds`、`runs` 和 `adapter-tmp` 在各节点必须指向同一共享存储，不能使用互不共享的节点本地目录。

```bash
export DATA_ROOT=data/rlds
export REASONING_DATASET_PATH="${DATA_ROOT}/libero_reasonings.json"
export VLA_PATH=openvla/openvla-7b
export RUN_ROOT_DIR=runs
export ADAPTER_TMP_DIR=adapter-tmp
export NODE_NUM=4
export NPROC_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR=192.0.2.1
export MASTER_PORT=29503
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

`192.0.2.1` 是文档示例地址，不是可用集群地址。`IMAGE_AUG=False` 沿用启动脚本默认值，不能据此认定论文采用相同增强设置。

think-then-act 实验将 `TRAIN_TYPE`、`RUN_ID_NOTE` 改为 `think_first`；其余控制变量保持一致。两个模式都监督完整回答。这里不提供 `TRAIN_TYPE=no_cot` 或 `use_reasoning=False` 的基线命令，因为当前代码未实现完整对应分支。

### 输出与保存

- `RUN_ROOT_DIR` 默认 `<仓库根目录>/runs`，adapter 临时目录默认 `<仓库根目录>/adapter-tmp`。
- `RUN_ID_NOTE` 默认与 `TRAIN_TYPE` 相同，进入实验名称，避免两个策略使用同名目录；重复同配置实验仍需设置新的运行标识。
- TensorBoard 位于对应训练运行目录的 `tensorboard/`。
- `SAVE_LATEST_ONLY=False` 时，LoRA 合并后的完整模型写入带 `--<step>_chkpt` 后缀的目录，并保存 processor 与数据统计量。adapter 目录不等于完整评估模型。
- 当前仅按保存间隔写 checkpoint；30,000 不能整除 200,000，故示例显式使用 `SAVE_STEPS=10000`。检查实际日志中的保存步数，不假定训练结束必然另存最终模型。

### 保留的全量训练和格式转换入口

`vla-scripts/launch_train.sh` 使用 Prismatic/FSDP，**不是论文附录 D.2 的 LoRA 主配方**。其默认学习率为 3e-5、每卡 batch 32、epochs 5、图像增强 True。`MAX_STEPS` 可通过环境变量覆盖已有 VLA 配置；未设置时按 epochs 配置。入口要求：

- `PRETRAINED_CHECKPOINT`：Prismatic 格式 checkpoint 文件，不是 Hugging Face 模型目录。
- `DATA_ROOT`、`REASONING_DATASET_PATH`：与上述接口一致。
- `EXPECTED_WORLD_SIZE`、`GLOBAL_BATCH_SIZE` 由启动器按设备配置计算。
- `TRACKERS` 默认 JSON 数组 `["tensorboard"]`；不得把凭据放进该配置。
- 该上游训练入口及转换入口默认读取本地 `.hf_token` 文件；仓库不分发该文件。需要时由用户安全地提供自己的未跟踪凭据文件，不能将真实 token 写入文档、命令日志或 Git。Hugging Face CLI 登录缓存不能自动代替这两个入口的显式文件读取。

[权重转换脚本](../vla-scripts/extern/convert_openvla_weights_to_hf.py) 需要训练目录中的配置与 `checkpoints/latest-checkpoint.pt`。`local_tokenizer_path` 如被设置，必须与训练 tokenizer 一致。`LLAMA2_7B_PATH` 可覆盖底层 Llama-2 模型路径，默认 `meta-llama/Llama-2-7b-hf`；该模型可能要求独立访问授权。

## 评估配置与统计

### checkpoint 前置条件

1. 明确 checkpoint 对应 `think_first` 还是 `answer_first`，并使用相同评估提示词。
2. 准备完整模型、正确的 processor/tokenizer 和模型配置。现有加载器可能执行 checkpoint 中的自定义代码，只使用可信资源。
3. 核实 `dataset_statistics.json` 与动作定义及训练数据一致。评估默认查找 `libero_90`，而训练注册名为 `libero_lm_90`；当前没有自动别名转换。
4. 先处理本文件列出的解码缺口，否则即使脚本完成运行，也不能据此认为动作是正确解析的。

### 单进程评估入口

从仓库根目录运行。下面使用无增强训练的示例配置；实际裁剪设置必须依据 checkpoint 的训练预处理决定。

```bash
export EVAL_TYPE=answer_first
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES=0
export ROLLOUT_DIR=rollouts/answer_first_seed7
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint checkpoints/answer-first-checkpoint \
  --task_suite_name libero_90 \
  --center_crop False \
  --num_trials_per_task 10 \
  --seed 7 \
  --local_log_dir logs/answer_first_seed7 \
  --run_id_note answer_first_seed7 \
  --use_wandb False
```

### 多 GPU 评估入口

```bash
export MODEL_PATH=checkpoints/answer-first-checkpoint
export EVAL_TYPE=answer_first
export TASK_SUITE=libero_90
export TOTAL_TASKS=90
export GPU_COUNT=8
export NUM_TRIALS_PER_TASK=10
export SEED=7
export CENTER_CROP=False
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LOG_DIR=logs/answer_first_seed7
export ROLLOUT_DIR=rollouts/answer_first_seed7
bash vla-scripts/launch_libero_eval.sh
```

- 每张 GPU 独立运行一个任务子集，不是 DDP 模型并行。当前脚本按 GPU ID `0..GPU_COUNT-1` 分配，只适用于这些 GPU 确实分配给当前作业的环境。
- `TOTAL_TASKS` 必须与任务套件一致，不能把默认 90 用于只有 10 个任务的其他套件。
- `task_start` 为包含端点，`task_end` 为不包含端点；完整 LIBERO-90 运行计划 900 个回合。
- 每回合最多 400 个动作步，另有默认 10 步稳定等待。seed 默认 7，是当前代码默认值，不代表论文所有重复实验的种子列表。
- `ENABLE_GPU_HOLD=0` 为默认值。设为 `1` 会额外创建周期性 GPU 计算进程，通常无需启用，也不属于模型推理。
- 日志默认按评估类型放在 `logs/<EVAL_TYPE>/`，视频默认放在 `rollouts/<EVAL_TYPE>/<日期>/`。同策略重复实验应覆盖日志和视频目录，避免混淆。
- 汇总成功率须用各进程最终成功数之和除以回合数之和，不能简单平均不同任务数进程的百分比。检查任务覆盖、异常日志和分母；不要只根据 Shell 完成提示判定成功。
- 当前论文描述的整回合零动作排除未实现。正式发布结果时须解释这一差异，并保留原始结果；不能静默排除失败回合以匹配论文数字。

## 配置变量速查

| 变量 | 用途与默认值 |
| --- | --- |
| `CODE_DIR` | 启动器所在仓库根目录，可覆盖；从仓库根目录启动，数据与模型示例使用相对路径 |
| `DATA_ROOT` | 训练数据根目录，启动器必填 |
| `REASONING_DATASET_PATH` | 启动器默认 `${DATA_ROOT}/libero_reasonings.json` |
| `VLA_PATH` | LoRA 基座，默认 `openvla/openvla-7b` |
| `TRAIN_TYPE` / `EVAL_TYPE` | 默认 `think_first`，另一个已有模式为 `answer_first` |
| `NPROC_PER_NODE` | 训练每节点 GPU 数；未设置时兼容 `GPU_NUM`，再默认 8 |
| `NODE_NUM` / `NODE_RANK` | 节点数量和索引；默认 1 / 0，rank 未设置时兼容已有 `RANK` |
| `RUN_ROOT_DIR` / `ADAPTER_TMP_DIR` | 默认仓库下 `runs/` / `adapter-tmp/` |
| `RUN_ID_NOTE` | LoRA 默认采用训练类型，评估运行标识包含评估类型 |
| `MAX_STEPS` | LoRA 默认 200,000；全量训练未设置时使用 epochs |
| `HF_HOME` | 遵循用户设置；全量启动器未设置时使用标准用户缓存目录 |
| `LLAMA2_7B_PATH` | 可选底层模型路径，默认公开 Llama-2 ID |
| `MODEL_PATH` | 多 GPU 评估的 checkpoint，必填 |
| `LOG_DIR` / `ROLLOUT_DIR` | 多 GPU 评估日志与视频根目录 |
| `NUM_TRIALS_PER_TASK` / `SEED` | 多 GPU 评估透传参数，默认 10 / 7 |
| `CENTER_CROP` | 多 GPU 启动器原有默认 True，必须与实际训练预处理核对 |
| `ENABLE_GPU_HOLD` | 默认 0；普通评估不需要保活计算 |
| `TMPDIR` | 可选保活脚本的临时目录根路径；未设置时使用仓库下 `.cache/` |

## 验证边界

静态检查只能验证语法、参数和文档引用，不能证明训练收敛、动作解析正确或仿真成功率。当前整理版本不附带“已复现 Table 6”的声明。进一步发布前的检查与尚未完成事项见 [release_checklist.md](release_checklist.md)。
