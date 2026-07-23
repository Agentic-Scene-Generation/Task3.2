# SceneExpert 中文快速复现指南

本文档面向无外网服务器集群，目标是在已经创建并激活 `.venv` 后，用本地 vLLM + ModelScope 模型运行 SceneExpert/SceneSmith 实验入口：

```bash
bash ./scripts/run_experiment.sh
```

## 1. 项目运行结构

SceneExpert 基于 SceneSmith 的室内场景生成管线开发。主入口是 `main.py`，配置由 Hydra 管理，默认实验会依次运行：

1. `floor_plan`：生成房间/户型几何。
2. `furniture`：放置家具。
3. `wall_mounted`：放置墙面物体。
4. `ceiling_mounted`：放置天花板物体。
5. `manipuland`：放置桌面、柜面等小物体。

SceneExpert 增加在 `ablation_3/4/4a/4b/4c/5` 中：

- `ablation_2_qwen3_naive`：只用 Qwen + 原 SceneSmith。
- `ablation_3_qwen3_harness`：启用 SceneExpert harness 和 stage brief。
- `ablation_4_qwen3_harness_memory`：旧版快速记忆 MVP 配置。
- `ablation_4a_qwen3_lexical_memory`：lexical 快速记忆消融。
- `ablation_4b_qwen3_vector_memory`：BGE-M3 + numpy 向量记忆消融。
- `ablation_4c_qwen3_hybrid_memory`：structured filter + vector recall + hybrid score 的推荐记忆配置。
- `ablation_5_qwen3_full`：用于 LoRA/合并模型版本，运行时核心机制同 memory 版本，区别是 served model。

`scripts/run_experiment.sh` 会在同一个 job 里启动 vLLM，然后运行：

```bash
python main.py experiment="$EXPERIMENT" +name="$RUN_NAME"
```

### 当前 SceneExpert 实现边界

当前主运行路径采用 `scenesmith/scene_expert/hooks.py` 的 hook runner，而不是单独调用 `SceneExpertPipeline`：

- SceneExpert 配置优先读 `experiment.scene_expert`，也就是 `ablation_3/4/4a/4b/4c/5` 中的配置；根级 `scene_expert` 是默认禁用的 fallback。
- `floor_plan` 阶段通过 `pre_floor_plan` 把 StageBrief 和 memory directives 注入原始 prompt；`furniture`、`wall_mounted`、`ceiling_mounted`、`manipuland` 四个房间级阶段则把 StageBrief 和 memory directives 追加到 `scene.text_description`。
- 核心 SceneSmith agent 循环没有被改写；PR-4 只增强 hook 层注入，确保检索到的成功经验、失败约束和技能文本能显式进入后续 stage prompt。
- 当前 hook 路径中的 RepairController 会记录修复建议、写入 trace/memory，但不会在同一次 hook 中真正重跑失败阶段。
- SceneExpert hook 是 per-scene、非线程安全对象。启用 SceneExpert 时，即使设置了 `experiment.pipeline.parallel_rooms=true`，代码也会自动退回顺序房间生成；跨 scene 的 `experiment.num_workers` 仍可用，但 memory 模式建议先用 `num_workers=1`，避免多个进程同时写同一个 JSONL memory。

## 2. 推荐硬件和系统环境

最省事的集群配置：

- Linux 节点，Python 3.11。
- CUDA 12.x，推荐 CUDA 12.4。
- GPU：至少 2 张 80GB GPU 运行 35B MoE 长上下文更稳。单卡可尝试降低上下文长度。
- 系统包：`git git-lfs wget unzip cmake build-essential libgl1 libegl1 libxrender1 libxkbcommon0 libsm6 libxext6 libxi6 libxxf86vm1 libglib2.0-0`。
- 可选：`bubblewrap`，多 GPU 并行渲染时可减少 Blender 抢 GPU。

无 sudo 的集群上，用管理员预装模块或容器提供这些系统依赖。

## 3. 统一可配置环境变量

这些参数可以直接在终端里 `export`，但更推荐写到一个 bash 配置文件中。项目已提供模板：

```bash
cp .env.example .env
vim .env
```

`scripts/run_experiment.sh`、`scripts/start_vllm.sh`、`scripts/deploy_qwen.sh` 会自动读取项目根目录的 `.env`。`.env` 已经被 `.gitignore` 忽略，适合保存每台服务器自己的路径、端口和模型设置。

注意：`cp .env.example .env` 只是生成配置文件，不会自动修改当前已经打开的终端环境。虚拟环境激活和 `.env` 加载也是两件事。如果想在当前终端里用 `echo` 检查变量，需要手动加载一次：

```bash
source .env
echo "$SCENEEXPERT_MODEL_ID"
```

修改 `.env` 后，如果还想让当前终端立即看到新值，也需要重新执行 `source .env`。直接运行项目脚本时不需要手动 source，脚本会自动加载。

如果集群上要为不同用户、模型或队列维护多份配置，也可以显式指定配置文件：

```bash
SCENEEXPERT_ENV_FILE=/share/configs/sceneexpert_qwen35.env \
  bash scripts/run_experiment.sh ablation_4_qwen3_harness_memory
```

`.env` 本质上是一个 bash 脚本，填写格式如下：

为什么这里不把它做成 YAML 主配置：Hydra YAML 适合 Python 内部配置，但 vLLM、`OPENAI_BASE_URL`、SLURM 作业环境需要在 `python main.py` 启动前就生效。用 `.env` 可以同时服务 shell、vLLM 和 Hydra，是当前最少转换、最显式的方式。

```bash
export SCENEEXPERT_MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export SCENEEXPERT_MODELS_DIR="/share/models"
export SCENEEXPERT_MODEL_DIR="/share/models/Qwen3.5-35B-A3B"
export SCENEEXPERT_HSSD_DATA_DIR="/share/datasets/hssd_hsm"
export SCENEEXPERT_DATA_DIR="/scratch/$USER/sceneexpert_data"
export SCENEEXPERT_OPENCLIP_DIR="$SCENEEXPERT_DATA_DIR/openclip"
export SCENEEXPERT_OPENCLIP_CHECKPOINT="$SCENEEXPERT_OPENCLIP_DIR/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin"
export SCENEEXPERT_REQUIRE_LOCAL_OPENCLIP=1
export SCENEEXPERT_CHECKPOINTS_DIR="/share/models/sam3d_checkpoints"
export SCENEEXPERT_OUTPUT_DIR="/scratch/$USER/sceneexpert_outputs"
export SCENEEXPERT_MEMORY_DIR="$SCENEEXPERT_OUTPUT_DIR/scene_expert_memory"

export SCENEEXPERT_VLLM_PORT=8000
export SCENEEXPERT_VLLM_HEALTH_URL="http://localhost:8000/health"

# 单卡快速复现建议先用保守配置。
export SCENEEXPERT_TENSOR_PARALLEL_SIZE=1
export SCENEEXPERT_MAX_MODEL_LEN=32768
export SCENEEXPERT_GPU_MEMORY_UTILIZATION=0.90
export SCENEEXPERT_GPU_PREFLIGHT_CHECK=1
export SCENEEXPERT_VLLM_DTYPE="auto"
export SCENEEXPERT_VLLM_QUANTIZATION=""
export SCENEEXPERT_VLLM_CPU_OFFLOAD_GB=20
export SCENEEXPERT_VLLM_KV_CACHE_DTYPE="auto"
export SCENEEXPERT_VLLM_WAIT_TIMEOUT_SECONDS=5400
export SCENEEXPERT_VLLM_ENGINE_READY_TIMEOUT_S=5400
export SCENEEXPERT_VLLM_SAFETENSORS_LOAD_STRATEGY="prefetch"
export SCENEEXPERT_VLLM_USE_DEEP_GEMM=0
export SCENEEXPERT_VLLM_MOE_USE_DEEP_GEMM=0
export SCENEEXPERT_VLLM_DEEP_GEMM_WARMUP="skip"
export SCENEEXPERT_VLLM_ENFORCE_EAGER=1

# 快速复现默认只使用 HSSD 静态资产。补齐 artvip_sdf 或 partnet_mobility_sdf 后再改为 0。
export SCENEEXPERT_DISABLE_ARTICULATED=1

# 如果没有 materials/ 和 materials/embeddings/，保持为 1，先跳过材料检索。
export SCENEEXPERT_DISABLE_MATERIALS=1

export SCENEEXPERT_ENABLE_AUTO_TOOL_CHOICE=1
export SCENEEXPERT_TOOL_CALL_PARSER="qwen3_xml"
export SCENEEXPERT_REASONING_PARSER="qwen3"
```

如果作业能看到 2 张或更多 GPU，再切到长上下文配置：

```bash
export SCENEEXPERT_TENSOR_PARALLEL_SIZE=2
export SCENEEXPERT_MAX_MODEL_LEN=262144
```

切换到其他开源模型时，只要保证 vLLM 的 `--served-model-name` 和 Hydra 的 `llm.model_id` 一致：

```bash
export SCENEEXPERT_MODEL_ID="Qwen/Qwen3.6-35B-A3B"
export SCENEEXPERT_MODEL_DIR="/share/models/Qwen3.6-35B-A3B"
```

项目配置会自动把所有 agent 的 `openai.model` 指向 `${llm.model_id}`。因此，推荐把“常用默认值”放进 `.env`，把“单次实验差异”放到命令行 Hydra override。

如果切到非 Qwen3 系列模型，需要同时确认 vLLM 的 tool parser/reasoning parser。比如某些模型需要关闭 reasoning parser：

```bash
export SCENEEXPERT_REASONING_PARSER=""
export SCENEEXPERT_TOOL_CALL_PARSER="hermes"
```

具体 parser 名称以当前 vLLM 版本支持为准。

## 4. Python 依赖安装

你已经完成：

```bash
source .venv/bin/activate
```

`run_experiment.sh` 默认也会激活项目根目录的 `.venv`。如果你实际使用的是 Conda 环境或其他虚拟环境，并且依赖已经装在那里，需要在 `.env` 中设置：

```bash
export SCENEEXPERT_ACTIVATE_VENV=0
```

如果只是项目虚拟环境不叫 `.venv`，可以指定路径：

```bash
export SCENEEXPERT_ACTIVATE_VENV=1
export SCENEEXPERT_VENV_PATH="/path/to/your/venv"
```

否则会出现一种很容易误判的情况：你在当前环境里安装了 `vllm`，但 `run_experiment.sh` 启动时又切换到另一个 `.venv`，于是脚本里仍然找不到 `vllm`。

`scripts/acp_sceneexpert.sh` 也会按同样规则在启动前激活 Python 环境，并在日志里打印：

```text
Activated Python env: ...
Python executable: ...
```

如果 ACP 日志显示 `ModuleNotFoundError: No module named 'FlagEmbedding'`，但你手动执行 `python -m pip install -r requirements-memory.txt` 时又提示已经安装，优先检查 ACP 日志中的 `Python executable` 是否指向同一个 `.venv/bin/python`。如果不是同一个环境，设置 `SCENEEXPERT_VENV_PATH`，或把 `SCENEEXPERT_ACTIVATE_VENV=0` 与提交任务时已经激活的 Conda/venv 环境配套使用。

先确认当前虚拟环境里有 `pip`：

```bash
python -m pip --version
```

如果报错：

```text
No module named pip
```

说明这个 `.venv` 创建成功了，但没有把 `pip` 装进去。优先用 Python 自带的 `ensurepip` 补齐：

```bash
python -m ensurepip --upgrade
python -m pip --version
```

如果集群能访问内网 PyPI 镜像，再升级基础打包工具：

```bash
python -m pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果没有任何镜像出口，不要在这里强行升级；保留 `ensurepip` 装出的基础 `pip`，后面用离线 wheelhouse 安装依赖。

如果 `python -m ensurepip --upgrade` 也失败，通常是系统 Python 没有安装 `venv/ensurepip` 组件，或当前 `.venv` 是用 `--without-pip` 创建的。优先让管理员提供带 `ensurepip` 的 Python 3.11 模块，或重建虚拟环境：

```bash
deactivate 2>/dev/null || true
mv .venv .venv.no-pip.bak
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip --version
```

完全无外网且 `ensurepip` 不可用时，需要在同系统、同 Python 版本的联网机器或内网构建节点提前准备 `get-pip.py` 和基础 wheel：

```bash
python -m pip download pip setuptools wheel -d wheelhouse_bootstrap
# 同时下载 https://bootstrap.pypa.io/get-pip.py，并与 wheelhouse_bootstrap 一起传到集群共享盘
```

然后在集群节点当前 `.venv` 中离线注入：

```bash
source .venv/bin/activate
python /share/get-pip.py --no-index --find-links /share/wheelhouse_bootstrap pip setuptools wheel
python -m pip --version
```

如果集群能访问内网 PyPI 镜像，直接安装：

```bash
python -m pip install -U pip
python -m pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --frozen --no-dev
python -m pip install modelscope vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install "numpy>=1.26,<2.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

注意：`vllm` 和 `modelscope` 是运行脚本需要的依赖，不在 `pyproject.toml` 主依赖里，需要单独装。`vllm` 安装过程可能把 NumPy 升级到 2.x，但 `bpy==4.5.4` / Blender 扩展通常按 NumPy 1.x ABI 编译；如果日志出现 `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`，必须重新执行上面的 NumPy pin 命令。`scripts/run_experiment.sh` 已加入预检查，发现 NumPy 2.x 会在启动 vLLM 前直接停止，避免浪费 30 分钟模型启动时间。

如果要运行向量 / hybrid memory 版本，还需要安装可选 memory 依赖。`requirements-memory.txt` 不是可执行脚本，而是 pip 的依赖清单；在项目根目录执行：

```bash
python -m pip install -r requirements-memory.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中 `-r requirements-memory.txt` 的意思是让 pip 按文件内容安装依赖。当前第一版只包含 `FlagEmbedding` 和 `numpy`，不包含 FAISS、reranker 或 ModelScope。只跑 `ablation_4a_qwen3_lexical_memory` 时不需要这一步；跑 `ablation_4b_qwen3_vector_memory` / `ablation_4c_qwen3_hybrid_memory`，或执行 `scripts/build_memory_index.py` 时需要。

完全无外网时，建议在同系统、同 Python 版本的联网机器或内网构建节点准备 wheel/cache：

```bash
# 联网或可访问内网镜像的构建节点
python -m pip install uv
uv sync --frozen --no-dev
python -m pip download modelscope vllm -d wheelhouse -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip download -r requirements-memory.txt -d wheelhouse_memory -i https://pypi.tuna.tsinghua.edu.cn/simple

# 打包 uv cache、wheelhouse、代码仓库后传到集群共享盘
```

集群节点安装：

```bash
source .venv/bin/activate
export UV_CACHE_DIR=/share/cache/uv_sceneexpert
uv sync --frozen --no-dev --offline
python -m pip install --no-index --find-links /share/wheelhouse modelscope vllm
python -m pip install --no-index --find-links /share/wheelhouse "numpy>=1.26,<2.0"
python -m pip install --no-index --find-links /share/wheelhouse_memory -r requirements-memory.txt
```

如果 `bpy==4.5.4` 无法离线解析，需要把 Blender PyPI 的对应 wheel 也预先放进 cache 或 wheelhouse。

## 5. ModelScope 下载本地大模型

集群有 ModelScope 出口时：

```bash
source .venv/bin/activate
export SCENEEXPERT_MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export SCENEEXPERT_MODEL_DIR="/share/models/Qwen3.5-35B-A3B"
modelscope download --model "$SCENEEXPERT_MODEL_ID" --local_dir "$SCENEEXPERT_MODEL_DIR"
```

集群完全无外网时，在可访问 ModelScope 的机器上下载，然后复制到共享盘同一路径。

也可以使用脚本下载并启动 vLLM：

```bash
SCENEEXPERT_INSTALL_RUNTIME_DEPS=1 bash scripts/deploy_qwen.sh
```

离线集群通常不建议让 `deploy_qwen.sh` 自动安装依赖，先手动从 wheelhouse 安装更可控。

## 6. 数据目录准备

最快复现建议使用默认 HSSD 检索资产，而不是 SAM3D/Hunyuan3D 生成资产。这样不需要在集群上安装 HuggingFace 版 SAM3D 模型链路。

推荐把只读 HSSD/HSM 数据和可写的 SceneExpert 扩展数据分开：

```text
$SCENEEXPERT_HSSD_DATA_DIR/       # 通常是共享盘只读目录，例如 /mnt/afs/task3_2/share_data/hsm
  hssd-models/
    support-surfaces/
  preprocessed/

$SCENEEXPERT_DATA_DIR/            # 用户可写目录，例如 /mnt/afs/task3_2/L202500276_lwz/data
  artvip_sdf/
    embeddings/
  materials/
    embeddings/
  partnet_mobility_sdf/
    embeddings/

$SCENEEXPERT_OPENCLIP_DIR/
  DFN5B-CLIP-ViT-H-14-378/
    open_clip_pytorch_model.bin
```

这些数据在原 SceneSmith README 中部分来自 HuggingFace、GitHub release 或 AmbientCG。无外网集群上不要直接运行 HuggingFace 下载命令，建议由数据管理员提前下载并放入共享盘。

### HSSD 资产库下载

HSSD-only 快速复现至少需要确认以下内容：

- `hssd-models/`：HuggingFace 上的 HSSD GLB 资产库，约 72GB。
- `preprocessed/`：HSM 提供的 HSSD CLIP 检索索引，约 60MB。
- `hssd-models/support-surfaces/`：HSM 提供的预计算支撑面，约 2GB。
- `$SCENEEXPERT_OPENCLIP_DIR/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin`：HSSD 文本检索使用的 OpenCLIP 文本编码器权重。注意：`preprocessed/clip_hssd_embeddings.npy` 只是资产库侧的预计算向量，不包含文本编码器权重；OpenCLIP 权重建议单独放，不放在 HSSD/HSM 数据根目录里。

先加载 `.env`，确认 HSSD 只读目录和用户可写目录：

```bash
cd /path/to/SceneExpert
source .env
echo "$SCENEEXPERT_HSSD_DATA_DIR"
echo "$SCENEEXPERT_DATA_DIR"
mkdir -p "$SCENEEXPERT_DATA_DIR"
```

注意：`/mnt/afs/task3_2/share_data/hsm` 这类共享 HSSD/HSM 目录通常不可写。不要把 materials、ArtVIP、OpenCLIP 下载到这里；它们应放到可写的 `$SCENEEXPERT_DATA_DIR`，例如 `/mnt/afs/task3_2/L202500276_lwz/data`。

如果执行 `git lfs install` 报：

```text
git: 'lfs' is not a git command
```

说明系统没有安装 Git LFS。它不是 Python 包，不能靠激活 `.venv` 解决。能使用系统包管理器时：

```bash
# Debian/Ubuntu 容器或节点
apt-get update
apt-get install -y git-lfs

# RHEL/CentOS 系列节点可用 dnf/yum，由集群环境决定
dnf install -y git-lfs
```

没有 sudo 时，优先使用管理员提供的模块、容器镜像，或 Conda/Mamba：

```bash
conda install -c conda-forge git-lfs
git lfs install
git lfs version
```

HuggingFace 可访问时，先在网页上接受 HSSD license，然后下载 HSSD 模型。推荐用 HTTPS，避免依赖 HuggingFace SSH key：

```bash
cd "$SCENEEXPERT_HSSD_DATA_DIR"
git lfs install
git clone https://huggingface.co/datasets/hssd/hssd-models
```

如果数据集需要登录，先运行：

```bash
huggingface-cli login
```

如果当前环境始终装不了 Git LFS，也可以用 `huggingface-cli` 下载，不依赖 `git lfs` 命令：

```bash
python -m pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
huggingface-cli login
huggingface-cli download hssd/hssd-models \
  --repo-type dataset \
  --local-dir "$SCENEEXPERT_HSSD_DATA_DIR/hssd-models"
```

然后下载 HSSD 检索索引和 support-surfaces。项目脚本会自动读取 `.env`，并写入 `$SCENEEXPERT_HSSD_DATA_DIR`：

```bash
cd /path/to/SceneExpert
bash scripts/download_hssd_data.sh
```

如果 `$SCENEEXPERT_HSSD_DATA_DIR` 在当前集群节点不可写，不要在该节点运行 HSSD 下载脚本；应由数据管理员在可写的数据构建节点准备好 HSSD/HSM 目录，然后以只读方式挂载到集群。最终目录必须保持：

```text
$SCENEEXPERT_HSSD_DATA_DIR/
  hssd-models/
    objects/
    support-surfaces/
  preprocessed/
    hssd_wnsynsetkey_index.json
    clip_hssd_embeddings.npy
    clip_hssd_embeddings_index.yaml
    object_categories.json

$SCENEEXPERT_OPENCLIP_DIR/
  DFN5B-CLIP-ViT-H-14-378/
    open_clip_pytorch_model.bin
```

复制完成后在集群节点验证：

```bash
source .env
test -d "$SCENEEXPERT_HSSD_DATA_DIR/hssd-models/objects"
test -d "$SCENEEXPERT_HSSD_DATA_DIR/hssd-models/support-surfaces"
test -f "$SCENEEXPERT_HSSD_DATA_DIR/preprocessed/hssd_wnsynsetkey_index.json"
test -f "$SCENEEXPERT_HSSD_DATA_DIR/preprocessed/clip_hssd_embeddings.npy"
test -f "$SCENEEXPERT_OPENCLIP_CHECKPOINT"
find "$SCENEEXPERT_HSSD_DATA_DIR/hssd-models/objects" -name '*.glb' -print -quit
```

### OpenCLIP 权重离线准备

HSSD 检索链路会用 OpenCLIP `ViT-H-14-378-quickgelu` + `dfn5b` 把文本查询编码到和 `preprocessed/clip_hssd_embeddings.npy` 相同的向量空间。无外网集群必须提前准备这个权重文件，否则会在家具检索时报：

```text
Failed to download weights for tag 'dfn5b'
open_clip_pytorch_model.bin
apple/DFN5B-CLIP-ViT-H-14-378
```

最简做法是在可访问 ModelScope 或内部模型镜像的机器上下载该模型目录，然后复制到共享盘。目标路径建议固定为：

```bash
source .env
mkdir -p "$SCENEEXPERT_OPENCLIP_DIR/DFN5B-CLIP-ViT-H-14-378"
modelscope download \
  --model apple/DFN5B-CLIP-ViT-H-14-378 \
  --local_dir "$SCENEEXPERT_OPENCLIP_DIR/DFN5B-CLIP-ViT-H-14-378"
```

如果 ModelScope 提示找不到这个模型 ID，请让数据管理员在 ModelScope/内网模型仓库中同步 `apple/DFN5B-CLIP-ViT-H-14-378`，或至少提供单文件 `open_clip_pytorch_model.bin`。最终只要这个文件存在即可：

```bash
ls -lh "$SCENEEXPERT_OPENCLIP_DIR/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin"
```

`.env` 中应显式写入：

```bash
export SCENEEXPERT_OPENCLIP_DIR="/mnt/afs/task3_2/L202500276_lwz/data/openclip"
export SCENEEXPERT_OPENCLIP_CHECKPOINT="${SCENEEXPERT_OPENCLIP_DIR}/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin"
export SCENEEXPERT_REQUIRE_LOCAL_OPENCLIP=1
```

`SCENEEXPERT_OPENCLIP_CHECKPOINT` 可以指向完整的 `.bin` 文件；如果指向模型目录，代码会自动在该目录下查找 `open_clip_pytorch_model.bin`。

项目现在会在运行前检查该文件，并在 HSSD retrieval server 启动时预加载 OpenCLIP。若文件缺失，会直接 fail fast，不再进入家具生成阶段反复调用 router，也不会继续尝试 table、box、sofa 等无意义补救。

### AmbientCG materials 材质库准备

materials 是可选数据。当前快速复现默认跳过它：

```bash
export SCENEEXPERT_DISABLE_MATERIALS=1
```

只有当你需要材质语义检索、地板/墙面 PBR 材质，或 `thin_covering` 相关能力时，才需要准备 `$SCENEEXPERT_DATA_DIR/materials`。这里的 `$SCENEEXPERT_DATA_DIR` 应该是用户可写目录，例如 `/mnt/afs/task3_2/L202500276_lwz/data`，不是只读的 `$SCENEEXPERT_HSSD_DATA_DIR`。

无外网集群不要直接运行下面的下载脚本，因为 `scripts/download_ambientcg.py` 和 `scripts/compute_ambientcg_embeddings.py` 都会访问 AmbientCG API/预览图。推荐在可联网的数据构建机完成下载和 embedding，然后把完整目录复制到集群共享盘。

在可联网机器上执行：

```bash
cd /path/to/SceneExpert
source .env

python scripts/download_ambientcg.py \
  --resolution 2K \
  --format JPG \
  --output "$SCENEEXPERT_DATA_DIR/materials" \
  --concurrent 8
```

如果下载阶段报：

```text
AttributeError: 'list' object has no attribute 'get'
```

说明 AmbientCG API 返回的 `downloadFiletypeCategories` 已从旧脚本预期的 dict 结构变成 list 结构，脚本没有解析到下载列表。请确认已经同步当前版本的 `scripts/download_ambientcg.py`；新版脚本已兼容 dict/list 两种返回结构。

下载完成后再计算检索 embedding：

```bash
python scripts/compute_ambientcg_embeddings.py \
  --materials-dir "$SCENEEXPERT_DATA_DIR/materials" \
  --output "$SCENEEXPERT_DATA_DIR/materials/embeddings" \
  --preview-size 1024 \
  --concurrent 8
```

如果只是小规模测试，可以先加 `--limit 100`。正式复现建议保留完整 `materials/` 和 `materials/embeddings/`。复制到集群后检查：

```bash
source .env
test -d "$SCENEEXPERT_DATA_DIR/materials"
test -f "$SCENEEXPERT_DATA_DIR/materials/embeddings/clip_embeddings.npy"
test -f "$SCENEEXPERT_DATA_DIR/materials/embeddings/embedding_index.yaml"
test -f "$SCENEEXPERT_DATA_DIR/materials/embeddings/metadata_index.yaml"
find "$SCENEEXPERT_DATA_DIR/materials" -maxdepth 2 -type f -name '*Color*' -print -quit
```

确认这些文件存在后，才在 `.env` 或 ACP 脚本中启用：

```bash
export SCENEEXPERT_DISABLE_MATERIALS=0
```

如果继续保持 `1`，SceneExpert 会跳过 materials retrieval，不影响 HSSD-only 的家具摆放复现。

### ArtVIP / PartNet articulated 资产准备

articulated 资产用于可开合柜门、抽屉、柜体等物体。当前快速复现默认跳过它：

```bash
export SCENEEXPERT_DISABLE_ARTICULATED=1
```

本项目支持两个 articulated source：

- `artvip_sdf/`：推荐使用，质量通常优于 PartNet-Mobility。原 SceneSmith 提供过预处理包 `artvip/artvip_vhacd.tar.gz` 和 `artvip/artvip_coacd.tar.gz`，其中 VHACD 版本碰撞几何更紧，优先推荐。
- `partnet_mobility_sdf/`：可从 PartNet-Mobility 原始 URDF 转换得到，但原 README 也说明其 mesh 和 joint 质量较低，适合作为补充源。

无外网集群上，ArtVIP 最简做法是让数据管理员在可联网机器或内网镜像中提前下载 SceneSmith 预处理包，然后把压缩包复制到集群可见位置，再解压到用户可写的 `$SCENEEXPERT_DATA_DIR/artvip_sdf`。例如压缩包已经位于 `/share/raw/artvip_vhacd.tar.gz`：

```bash
source .env
mkdir -p "$SCENEEXPERT_DATA_DIR/artvip_sdf"
tar xzf /share/raw/artvip_vhacd.tar.gz -C "$SCENEEXPERT_DATA_DIR/artvip_sdf"
```

如果数据构建机可以访问 HuggingFace，原始下载命令是：

```bash
huggingface-cli download nepfaff/scenesmith-preprocessed-data \
  artvip/artvip_vhacd.tar.gz \
  --repo-type dataset \
  --local-dir /share/raw
```

下载完成后压缩包通常位于 `/share/raw/artvip/artvip_vhacd.tar.gz`，再复制或移动到集群可见的位置。CoACD 版本也可用：

```bash
huggingface-cli download nepfaff/scenesmith-preprocessed-data \
  artvip/artvip_coacd.tar.gz \
  --repo-type dataset \
  --local-dir /share/raw
```

如果压缩包已经包含 `embeddings/`，无需重新计算；如果没有 embeddings，或者你自己转换了 ArtVIP SDF，需要生成索引：

```bash
python scripts/compute_articulated_embeddings.py \
  --source artvip \
  --data-path "$SCENEEXPERT_DATA_DIR/artvip_sdf" \
  --output-path "$SCENEEXPERT_DATA_DIR/artvip_sdf/embeddings"
```

ArtVIP 目录应类似：

```text
$SCENEEXPERT_DATA_DIR/artvip_sdf/
  large_furniture/
    <model_id>/
      *.sdf
      *_properties.json
  small_furniture/
    <model_id>/
      *.sdf
      *_properties.json
  embeddings/
    clip_embeddings.npy
    embedding_index.yaml
    metadata_index.yaml
```

如果要使用 PartNet-Mobility，则先从数据源获取原始 `partnet-mobility-v0`，再转换为 SDF：

```bash
source .env
bash scripts/convert_partnet_parallel.sh \
  /share/raw/partnet-mobility-v0 \
  "$SCENEEXPERT_DATA_DIR/partnet_mobility_sdf" \
  8

python scripts/compute_articulated_embeddings.py \
  --source partnet_mobility \
  --data-path "$SCENEEXPERT_DATA_DIR/partnet_mobility_sdf" \
  --output-path "$SCENEEXPERT_DATA_DIR/partnet_mobility_sdf/embeddings"
```

复制或计算完成后检查：

```bash
source .env
test -f "$SCENEEXPERT_DATA_DIR/artvip_sdf/embeddings/clip_embeddings.npy"
test -f "$SCENEEXPERT_DATA_DIR/artvip_sdf/embeddings/embedding_index.yaml"
test -f "$SCENEEXPERT_DATA_DIR/artvip_sdf/embeddings/metadata_index.yaml"
find "$SCENEEXPERT_DATA_DIR/artvip_sdf" -name '*.sdf' -print -quit
find "$SCENEEXPERT_DATA_DIR/artvip_sdf" -name '*_properties.json' -print -quit
```

确认 ArtVIP 或 PartNet-Mobility 的 `embeddings/` 可用后，才在 `.env` 或 ACP 脚本中启用：

```bash
export SCENEEXPERT_DISABLE_ARTICULATED=0
```

如果继续保持 `1`，SceneExpert 会跳过 articulated retrieval，不影响 HSSD-only 的静态家具复现。

HSSD-only 快速复现的最小必需项：

- `$SCENEEXPERT_HSSD_DATA_DIR/hssd-models/`：HSSD 资产库。
- `$SCENEEXPERT_HSSD_DATA_DIR/preprocessed/`：HSSD CLIP 检索索引。
- `$SCENEEXPERT_OPENCLIP_DIR/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin`：HSSD 文本检索编码器权重。
- `$SCENEEXPERT_HSSD_DATA_DIR/hssd-models/support-surfaces/`：预计算支撑面，能显著加速。

完整能力的可选数据：

- `$SCENEEXPERT_DATA_DIR/materials/` 和 `$SCENEEXPERT_DATA_DIR/materials/embeddings/`：AmbientCG 材质与检索索引。
- `$SCENEEXPERT_DATA_DIR/artvip_sdf/` 和 `$SCENEEXPERT_DATA_DIR/artvip_sdf/embeddings/`：可开合柜门/抽屉等 articulated 资产。

如果暂时没有 SAM3D 权重，可以保持默认：

```yaml
asset_manager.general_asset_source: hssd
```

只有当你要使用生成式 3D 资产时，才需要准备：

```text
$SCENEEXPERT_CHECKPOINTS_DIR/
  sam3.pt
  pipeline.yaml
  ...
```

并设置：

```bash
furniture_agent.asset_manager.general_asset_source=generated
manipuland_agent.asset_manager.general_asset_source=generated
```

无 HuggingFace 环境下，SAM3D 权重同样需要离线预下载。

## 7. 启动 vLLM

SceneExpert 当前通过 OpenAI-compatible HTTP 接口调用大模型。`vLLM` 的作用是把本地 Qwen 模型目录服务化成这个接口，因此它不是 OpenAI 云服务，也不会使用真实 OpenAI key。`.env` 中的：

```bash
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:8000/v1"
```

表示“使用本机 vLLM 服务，key 只是占位符”。

如果只有本地模型文件，例如 `$SCENEEXPERT_MODEL_DIR`，但没有任何推理服务，就不能直接跳过 vLLM；agent 无法直接读取模型权重。你可以跳过的是“由 `run_experiment.sh` 自动启动 vLLM”这一步，前提是你已经用 vLLM、SGLang、llama.cpp server、TGI 等启动了一个兼容 OpenAI `/v1/chat/completions` 的本地服务，并且模型名与 `SCENEEXPERT_MODEL_ID` 一致。

单独启动服务：

```bash
source .venv/bin/activate
bash scripts/start_vllm.sh
```

确认：

```bash
curl http://localhost:8000/health
```

`run_experiment.sh` 默认会自动启动 vLLM。如果你已经单独启动了服务：

```bash
export SCENEEXPERT_START_VLLM=0
export OPENAI_BASE_URL="http://localhost:8000/v1"
export SCENEEXPERT_VLLM_HEALTH_URL="http://localhost:8000/health"
```

如果已有服务没有 `/health`，但支持 OpenAI 风格的模型列表接口，也可以把健康检查指向：

```bash
export SCENEEXPERT_VLLM_HEALTH_URL="http://localhost:8000/v1/models"
```

如果报 `vllm: command not found`，说明当前虚拟环境没有安装 vLLM，或者 `vllm` 命令不在 `PATH` 中。先检查：

```bash
which vllm
which python
python -c "import sys; print(sys.executable)"
python -m pip show vllm
python -c "import vllm; print(vllm.__version__); print(vllm.__file__)"
python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
```

如果你用的是 Conda 环境，并不想让脚本切换到项目 `.venv`，在 `.env` 中设置：

```bash
export SCENEEXPERT_ACTIVATE_VENV=0
```

能访问内网 PyPI 镜像时安装：

```bash
python -m pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

完全离线时，从 wheelhouse 安装：

```bash
python -m pip install --no-index --find-links /share/wheelhouse vllm
```

如果 `python -m pip show vllm` 能看到包，但 `which vllm` 仍然为空，通常是 console script 没写入当前 venv 的 `bin/`，或 shell 没刷新 `PATH`。先尝试：

```bash
hash -r
python -m pip install --force-reinstall --no-cache-dir vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果集群不能重装，但 `python -c "import vllm"` 成功，项目脚本现在会自动退回到模块入口：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model "$SCENEEXPERT_MODEL_DIR" \
  --served-model-name "$SCENEEXPERT_MODEL_ID" \
  --port "$SCENEEXPERT_VLLM_PORT"
```

也就是说，`which vllm` 为空不一定致命；关键是当前 `python` 能否导入 `vllm.entrypoints.openai.api_server`。

如果当前节点只有 1 张可见 GPU，而 `.env` 中写了 `SCENEEXPERT_TENSOR_PARALLEL_SIZE=2`，vLLM 也会启动失败。要么申请 2 张 GPU，要么改成：

```bash
export SCENEEXPERT_TENSOR_PARALLEL_SIZE=1
export SCENEEXPERT_MAX_MODEL_LEN=65536
```

## 8. 一键运行实验

可用实验名：

| 实验名 | 用途 | 是否建议当前运行 |
| --- | --- | --- |
| `indoor_scene_generation` | 默认室内场景生成实验入口，偏原始默认配置。 | 一般不作为对比实验首选，调试默认流程时可用。 |
| `base_experiment` | ablation 配置继承的基础配置。 | 不建议直接作为正式实验运行。 |
| `ablation_1_scenesmith_original` | 原 SceneSmith + GPT-4o 风格 baseline，不启用 Qwen/SceneExpert。 | 不建议用于当前本地 Qwen 复现，除非要对齐原论文 baseline。 |
| `ablation_2_qwen3_naive` | Qwen3.5 + 原 SceneSmith 流程，不启用 SceneExpert harness/memory。 | 建议先跑，作为最小 baseline 和环境冒烟测试。 |
| `ablation_3_qwen3_harness` | Qwen3.5 + SceneExpert harness + StageBrief，不启用 memory。 | 建议第二个跑，用来验证 SceneExpert hook 和 trace。 |
| `ablation_4_qwen3_harness_memory` | Qwen3.5 + harness + StageBrief + 旧版 fast memory。 | 保留作兼容配置；正式 memory 对比优先使用 4a/4b/4c。 |
| `ablation_4a_qwen3_lexical_memory` | Qwen3.5 + harness + lexical fast memory。 | memory 消融实验；不需要 BGE-M3 或 numpy index。 |
| `ablation_4b_qwen3_vector_memory` | Qwen3.5 + harness + BGE-M3 numpy vector memory。 | memory 消融实验；需要先构建 memory index。 |
| `ablation_4c_qwen3_hybrid_memory` | Qwen3.5 + harness + structured filter + vector recall + hybrid score。 | 推荐的增强 memory 实验；需要先构建 memory index。 |
| `ablation_5_qwen3_full` | harness + memory + LoRA/合并后的专用模型。 | 只有 LoRA/合并模型已经准备好并由 vLLM served 时再跑。 |

推荐顺序：

1. `ablation_2_qwen3_naive`：确认 vLLM、HSSD、基础 pipeline 能跑通。
2. `ablation_3_qwen3_harness`：确认 SceneExpert trace/harness 生效。
3. `ablation_4a_qwen3_lexical_memory`：确认 memory JSONL 读写和 lexical 检索可用。
4. `ablation_4b_qwen3_vector_memory`：验证 BGE-M3 embedding + numpy index。
5. `ablation_4c_qwen3_hybrid_memory`：正式 memory 增强实验。
6. `ablation_5_qwen3_full`：仅在 LoRA/合并模型完成后运行。

默认运行 Qwen naive baseline：

```bash
source .venv/bin/activate
bash scripts/run_experiment.sh ablation_2_qwen3_naive
```

运行 SceneExpert memory 版本：

```bash
bash scripts/run_experiment.sh ablation_4_qwen3_harness_memory
```

运行 PR-4 拆分后的 memory 消融版本：

```bash
# lexical memory，不需要向量索引
bash scripts/run_experiment.sh ablation_4a_qwen3_lexical_memory

# vector / hybrid memory，运行前先按第 9 节构建 numpy index
bash scripts/run_experiment.sh ablation_4b_qwen3_vector_memory
bash scripts/run_experiment.sh ablation_4c_qwen3_hybrid_memory
```

运行 full/LoRA 合并模型版本：

```bash
export SCENEEXPERT_MODEL_ID="Qwen/Qwen3-SceneExpert-LoRA"
export SCENEEXPERT_MODEL_DIR="/share/models/Qwen3-SceneExpert-LoRA"
bash scripts/run_experiment.sh ablation_5_qwen3_full
```

ACP 多卡提交：

开发机单卡 H100 80GB 加载未量化 Qwen3.5-35B-A3B 仍可能在 `Qwen3NextSparseMoeBlock` / `FusedMoE` / `torch.empty` 阶段失败。正式复现建议通过 ACP 申请多卡，并使用项目提供的 ACP 入口：

```bash
bash scripts/acp_sceneexpert.sh
```

ACP 相关参数不要写在终端前缀里，也不建议再写进 `.env`。推荐直接改 `scripts/acp_sceneexpert.sh` 顶部 `# TODO` 配置区，让 ACP 脚本成为多卡作业参数的唯一入口。2 张 H100 80GB 推荐：

```bash
ACP_EXPERIMENT="ablation_4c_qwen3_hybrid_memory"  # 未构建 index 时先用 ablation_4a_qwen3_lexical_memory
ACP_GPUS=2
ACP_CUDA_VISIBLE_DEVICES=""
ACP_MAX_MODEL_LEN=65536
ACP_GPU_MEMORY_UTILIZATION=0.90
ACP_CPU_OFFLOAD_GB=0
ACP_VLLM_WAIT_TIMEOUT_SECONDS=7200
ACP_VLLM_ENGINE_READY_TIMEOUT_S=7200
ACP_SAFETENSORS_LOAD_STRATEGY="prefetch"
ACP_VLLM_USE_DEEP_GEMM=0
ACP_VLLM_MOE_USE_DEEP_GEMM=0
ACP_VLLM_DEEP_GEMM_WARMUP="skip"
ACP_VLLM_ENFORCE_EAGER=1
ACP_DISABLE_ARTICULATED=1
ACP_DISABLE_MATERIALS=1
ACP_CONVEX_READY_TIMEOUT=180
ACP_CONVEX_MAX_OMP_THREADS=32
ACP_SCENE_WORKERS=1
ACP_SCENE_RETRY_ATTEMPTS=1
ACP_MP_START_METHOD="forkserver"
```

如果 ACP 申请 4 张 H100 80GB，可以把脚本 TODO 区改成：

```bash
ACP_GPUS=4
ACP_CUDA_VISIBLE_DEVICES=""
ACP_MAX_MODEL_LEN=131072
ACP_CPU_OFFLOAD_GB=0
ACP_VLLM_WAIT_TIMEOUT_SECONDS=7200
ACP_VLLM_ENGINE_READY_TIMEOUT_S=7200
ACP_SAFETENSORS_LOAD_STRATEGY="prefetch"
ACP_VLLM_USE_DEEP_GEMM=0
ACP_VLLM_MOE_USE_DEEP_GEMM=0
ACP_VLLM_DEEP_GEMM_WARMUP="skip"
ACP_VLLM_ENFORCE_EAGER=1
ACP_DISABLE_ARTICULATED=1
ACP_DISABLE_MATERIALS=1
ACP_CONVEX_READY_TIMEOUT=180
ACP_CONVEX_MAX_OMP_THREADS=32
ACP_GPU_MEMORY_UTILIZATION=0.90
ACP_SCENE_WORKERS=1
ACP_SCENE_RETRY_ATTEMPTS=1
ACP_MP_START_METHOD="forkserver"
```

参数原则：

- `ACP_GPUS` 要和 ACP 申请的 GPU 数一致。调度器管理的 ACP 作业里推荐保持 `ACP_CUDA_VISIBLE_DEVICES=""`，不要手动写 `0,1` 或 `0,1,2,3`，否则可能绕过调度器分配，误用到已有进程的物理 GPU。
- H100 80GB 上 `ACP_GPU_MEMORY_UTILIZATION=0.90` 表示 vLLM 每卡目标占用约 72GB、预留约 8GB；当前 ACP 默认保持该值。
- 2 卡优先用 `max_model_len=65536` 跑通；需要更长上下文时再升到 `131072`。
- 4 卡可尝试 `131072`，确认稳定后再尝试 `262144`。
- 多卡优先不开 CPU offload；如果 2 卡仍在模型加载阶段失败，再把脚本 TODO 区的 `ACP_CPU_OFFLOAD_GB` 改成 `10`。
- AFS/FUSE 上首次加载 35B MoE 模型可能超过 30 分钟；ACP 任务默认用 `ACP_VLLM_WAIT_TIMEOUT_SECONDS=7200`，不要让外层脚本因为首次编译慢就提前杀掉。
- vLLM 自己还有内部 engine ready 超时，默认只有 600 秒；ACP 任务默认用 `ACP_VLLM_ENGINE_READY_TIMEOUT_S=7200`，避免 `Timed out waiting for engine core processes to start`。
- `scripts/run_experiment.sh` 在等待 vLLM `/health` 时会每 60 秒打印一次心跳和 vLLM 最新日志行，避免 ACP 平台因为控制台长时间没有输出而自动结束任务。
- 如果主机内存充足，保持 `ACP_SAFETENSORS_LOAD_STRATEGY="prefetch"`，可以减少 safetensors 分片在网络文件系统上的懒加载等待。
- 当前离线环境没有兼容的 DeepGEMM 后端，保持 `ACP_VLLM_USE_DEEP_GEMM=0`、`ACP_VLLM_MOE_USE_DEEP_GEMM=0`、`ACP_VLLM_DEEP_GEMM_WARMUP="skip"`，并设置 `ACP_VLLM_ENFORCE_EAGER=1`。当前日志显示仅禁用 DeepGEMM 环境变量仍可能进入 `deep_gemm_warmup`；`--enforce-eager` 是本环境下更直接的稳定规避路线。
- 如果当前可写的 `SCENEEXPERT_DATA_DIR` 还没有 `artvip_sdf/` 或 `partnet_mobility_sdf/`，保持 `ACP_DISABLE_ARTICULATED=1`。脚本会自动关闭四个 agent 的 articulated 策略，避免启动 articulated retrieval server 后因没有任何可用 source 失败。
- 如果当前可写的 `SCENEEXPERT_DATA_DIR` 还没有 `materials/` 或 `materials/embeddings/`，保持 `ACP_DISABLE_MATERIALS=1`。脚本会自动关闭 floor-plan 材料检索和四个 agent 的 `thin_covering` 策略，避免启动 materials retrieval server 后失败。
- 如果日志出现 `Convex decomposition server did not become ready within 10.0s`，优先保持 `ACP_CONVEX_READY_TIMEOUT=180`、`ACP_CONVEX_MAX_OMP_THREADS=32`。该服务用于生成碰撞几何，不需要额外模型或数据；它依赖 Python 包 `coacd`、`vhacdx`、`trimesh` 和 `flask`，这些已在项目依赖中声明。
- `ACP_SCENE_WORKERS` 控制完整 task 的并发数。Qwen3.5-35B-A3B 的 TP=4 表示一个 vLLM 副本占用四张卡，不等于四个 task 各占一张卡；多个 task 会共享同一个 vLLM endpoint。`ablation_4b/4c` 默认保持 `1`，避免并发写公共 memory bank。`ablation_2/3` 可先试 `2`，验证显存和渲染稳定后再增加。
- `ACP_SCENE_RETRY_ATTEMPTS=1` 只对 `SIGSEGV`、`SIGABRT`、本地 vLLM timeout、连接中断等故障执行一次整场景重试。每次重试都会使用新的干净 worker 进程，失败的半成品保留在 `<run>/failed_attempts/`。
- Linux ACP 保持 `ACP_MP_START_METHOD="forkserver"`。普通 `fork` 会继承 CUDA/Drake/SQLite 的原生状态；`spawn` 会重新执行 `main.py`，并可能因 Blender 私有模块 `_bpy` 无法在子进程启动阶段加载而失败。项目使用不预加载 `__main__` 的干净 forkserver。
- 即使 `ACP_SCENE_WORKERS=1`，每个 prompt 也会在独立的新进程中执行，不会让第二个 task 继承第一个 task 的 CUDA、Drake、SQLite、OpenMP 或 Agents SDK 状态。
- ACP 在启动 vLLM 前会执行 bpy-free worker import 检查。日志必须出现 `Python preflight passed (scene worker is bpy-free)`；如果普通 scene worker 意外导入了 `bpy`、`BlenderRenderer` 或 `BlenderRenderApp`，脚本会立即停止，避免等待大模型加载完成后才失败。

每个场景目录会生成：

```text
scene_001/
  scene_status.json   # running / failed / completed、attempt、错误摘要
  _SUCCESS            # 仅完整跑到配置的 stop_stage 后生成
```

批量任务结束后，不能只按“目录存在”判断成功；应检查每个 `scene_NNN/_SUCCESS`。如果瞬态崩溃触发了重试，可在 `failed_attempts/scene_NNN_attempt_*/` 查看原始日志和 partial trace。

优先级规则：ACP 脚本会先复制当前 `.env`，再生成本次 job 专用 env 文件，并在文件末尾追加 `SCENEEXPERT_TENSOR_PARALLEL_SIZE`、`SCENEEXPERT_MAX_MODEL_LEN`、`SCENEEXPERT_GPU_MEMORY_UTILIZATION`、`SCENEEXPERT_VLLM_CPU_OFFLOAD_GB` 等多卡覆盖项。因此最终运行时，ACP 脚本生成的覆盖项优先级更高；`.env` 只保留模型目录、数据目录、输出目录、端口等服务器固定配置。

如果你的 `.env` 是从旧版模板复制的，里面已有 `SCENEEXPERT_ACP_*` 字段，可以直接删除；新版 `scripts/acp_sceneexpert.sh` 不再依赖这些字段。

SLURM：

```bash
sbatch scripts/run_experiment.sh ablation_4_qwen3_harness_memory
```

`run_experiment.sh` 的第一个参数是实验名，后面的参数会原样传给 Hydra。例如只跑到家具阶段：

```bash
bash scripts/run_experiment.sh ablation_4_qwen3_harness_memory \
  experiment.pipeline.stop_stage=furniture
```

使用 CSV prompt：

```bash
bash scripts/run_experiment.sh ablation_4_qwen3_harness_memory \
  experiment.csv_path=/share/prompts/prompts.csv
```

降低显存压力：

```bash
export SCENEEXPERT_MAX_MODEL_LEN=65536
export SCENEEXPERT_GPU_MEMORY_UTILIZATION=0.85
bash scripts/run_experiment.sh ablation_2_qwen3_naive experiment.num_workers=1
```

输出目录默认是：

```text
$SCENEEXPERT_OUTPUT_DIR/YYYY-MM-DD/HH-MM-SS/
```

每次运行会保存 `resolved_config.yaml`，便于复现。

### 将 `.blend` 渲染为 PNG 检查图

SceneExpert 最终或中间结果里可能只有 `house.blend`，但没有对应的 Blender 渲染图。这个转换可以用 Python 脚本实现，不过必须用 **Blender 自带的 Python** 执行；普通虚拟环境里的 `python scripts/render_blend_views.py` 不能直接导入 `bpy`。

先确认服务器上有 Blender 命令：

```bash
which blender
blender --version
```

如果集群使用 module 管理软件，通常需要先执行类似命令：

```bash
module avail blender
module load blender/4.2
```

如果没有 module，需要让管理员安装 Blender，或在有外网机器下载 Blender Linux 压缩包后上传到集群并把 `blender` 加入 `PATH`。无 GUI 服务器也可以运行，关键是使用 `-b` 后台模式。

渲染一个 `.blend` 文件的推荐命令：

```bash
blender -b tmp/house.blend \
  --python scripts/render_blend_views.py -- \
  --output tmp/house_blend_views \
  --resolution 1024 \
  --engine eevee \
  --views top,north,east,south,west,iso
```

也可以不把 `.blend` 放在 `blender -b` 后面，而是显式传给脚本：

```bash
blender -b \
  --python scripts/render_blend_views.py -- \
  --input tmp/house.blend \
  --output tmp/house_blend_views
```

输出目录会包含多张 PNG，例如：

```text
tmp/house_blend_views/
  00_top.png
  01_north.png
  02_east.png
  03_south.png
  04_west.png
  05_iso.png
  render_manifest.json
```

如果顶视图被天花板遮挡，可以加：

```bash
--hide-ceiling
```

或者按名称隐藏任意对象：

```bash
--hide-name-contains ceiling --hide-name-contains roof
```

### 为 critic probe 批量生成最终高清视图

对一个运行目录执行：

```bash
blender -b --python scripts/render_critic_final_views.py -- \
  outputs/critic_probe/<run_id>
```

脚本会在每个 `scene_*/` 目录下生成 `critic_final_views/00_top.png` 和
`critic_final_views/01_side.png`（默认 2048 分辨率、无标签；side 视角只隐藏
朝向相机的墙）。共享 base 默认跳过；明确处理共享 base 时加
`--include-shared-base`。新运行也可以设置
`CRITIC_PROBE_RENDER_FINAL_VIEWS=true`，由
`scripts/run_parallel_critic_on.sh` 在所有 batch 完成后自动生成。

如果渲染太慢，优先使用 `--engine eevee`；只需要快速看几何布局时可改用 `--engine workbench`。如果服务器完全没有 Blender，只能在本地或图形节点打开 `.blend`：`File -> Open` 打开文件，设置相机视角后执行 `Render -> Render Image`，再通过 `Image -> Save As` 保存 PNG。

## 9. SceneExpert trace 与 memory

启用 `ablation_3/4/4a/4b/4c/5` 后，每个 experiment 输出目录会生成全局 trace；同时每个单场景目录会生成一个更适合调试和展示的 `scene_expert/` 目录：

```text
$SCENEEXPERT_OUTPUT_DIR/YYYY-MM-DD/HH-MM-SS/
  resolved_config.yaml
  experiment.log
  traces/
    trace_000000.json
  scene_000/
    timing_stats.jsonl
    stage_working_memory/
      furniture/
        memory.jsonl
      wall_mounted/
        memory.jsonl
    scene_expert/
      trace/
        trace_000000.json
        trace_000000_partial.json
      stages/
        000_floor_plan_pre.json
        001_floor_plan.json
        stage_trace.jsonl
      memory/
        memory_update_ops.json
        memory_update_ops.jsonl
      visuals/
        floor_plan_visuals.json
        furniture_visuals.json
```

说明：

- `trace/trace_000000_partial.json` 会在每个 stage 结束后刷新；即使任务中途报错，也会保留已完成 stage 的结构化记录。
- `stages/*_pre.json` 保存该 stage 开始前的 retrieved memory、StageBrief 和注入信息，便于检查 memory 是否真正进入 prompt。
- `stages/NNN_<stage>.json` 与 `stage_trace.jsonl` 保存该 stage 的 verifier report、scores、repair decision 和耗时。
- `memory/memory_update_ops.jsonl` 是本次场景结束时 MemoryWriter 生成的更新操作镜像，用于 debug；真正长期记忆仍写入下面的全局 memory bank。
- `visuals/*_visuals.json` 不重复拷贝大图，而是索引该 stage 已生成的 PNG、scores、scene_state、DMD 路径，便于快速定位可视化结果。
- `stage_working_memory/<stage>/memory.jsonl` 是单场景、单 stage 的在线工作记忆。每次 render 会写入一条记录；critic 评分后会把 scores 和 critique 追加到同一类记录中；下一次 designer 调用前会检索这些记录并注入 compact memory context。
- `timing_stats.jsonl` 记录 designer、critic、rendering_manager 等模块耗时；日志中也会打印 `[Timing]` 和 `[SceneExpertTiming]` 行，便于定位慢模块。

`ablation_4/4a/4b/4c/5` 会把长期经验写到各自的 memory 子目录，例如 `ablation_4c` 默认是：

```text
$SCENEEXPERT_MEMORY_DIR/ablation_4c/
  success_cases.jsonl
  failure_cases.jsonl
  skills.jsonl
```

这三个 JSONL 文件在 memory store 初始化时就会创建。文件存在但为空，表示当前还没有通过质量门控写入长期 memory；不是路径错误。

推荐流程：

1. 先跑 `ablation_3_qwen3_harness` 生成若干 trace。
2. 启动 vLLM。
3. 用 trace 初始化 `ablation_4` 的 memory。
4. 再跑 `ablation_4_qwen3_harness_memory`。

用 Qwen/MemoryWriter 从 trace 生成 memory：

```bash
python scripts/bootstrap_memory_from_traces.py \
  --traces-dir "$SCENEEXPERT_OUTPUT_DIR" \
  --memory-dir "$SCENEEXPERT_MEMORY_DIR/ablation_4"
```

如果只想从已有 `experiment.log` 中提取高分 placement，不调用 Qwen，可用：

```bash
python scripts/parse_log_to_memory.py \
  --log /path/to/experiment.log \
  --memory-dir "$SCENEEXPERT_MEMORY_DIR/ablation_4" \
  --scene-states-dir /path/to/scene_000/room_xxx/scene_states
```

### 构建 memory embedding 索引

当前运行主路径仍默认使用 lexical memory retriever；下面的步骤只是在 PR-2 阶段提前准备本地向量索引，供后续 hybrid retriever 使用。

BGE-M3 已按本项目约定下载到：

```bash
export SCENEEXPERT_MEMORY_EMBEDDING_MODEL_ID="BAAI/bge-m3"
export SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR="${SCENEEXPERT_MODELS_DIR}/bge-m3"
```

注意：`model_id` 只是语义标识，实际加载必须使用 `SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR`，不要写成 `${SCENEEXPERT_MODELS_DIR}/BAAI/bge-m3`。

安装可选 memory embedding 依赖：

```bash
# requirements-memory.txt 是 pip 依赖清单，不是 shell 脚本
python -m pip install -r requirements-memory.txt
```

如果集群只能访问内网镜像，使用：

```bash
python -m pip install -r requirements-memory.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

完全离线时，按第 4 节提前准备 `wheelhouse_memory/`，然后执行：

```bash
python -m pip install --no-index --find-links /share/wheelhouse_memory -r requirements-memory.txt
```

确认本地模型目录存在：

```bash
test -d "$SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR"
```

为 `ablation_4` 的 JSONL memory 构建 numpy index：

```bash
python scripts/build_memory_index.py \
  --memory-dir "$SCENEEXPERT_MEMORY_DIR/ablation_4" \
  --embedding-model-dir "$SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR" \
  --index-backend numpy \
  --device cpu
```

如果是在离线构建任务中且 GPU 空闲，可以把最后一行改成 `--device cuda`。在线运行阶段建议继续保持：

```bash
export SCENEEXPERT_MEMORY_EMBEDDING_DEVICE="cpu"
export SCENEEXPERT_MEMORY_INDEX_BACKEND="numpy"
```

构建完成后会生成：

```text
$SCENEEXPERT_MEMORY_DIR/ablation_4/indexes/
  success_furniture.npy
  success_furniture.metadata.jsonl
  success_furniture.manifest.json
  ...
```

构建完成并确认索引齐全后，可以在 `.env` 中启用 hybrid memory retrieval：

```bash
export SCENEEXPERT_MEMORY_RETRIEVER_TYPE="hybrid"
export SCENEEXPERT_MEMORY_INDEX_BACKEND="numpy"
export SCENEEXPERT_MEMORY_INDEX_REQUIRE_READY=true
export SCENEEXPERT_MEMORY_INDEX_AUTO_BUILD_MISSING=true
```

`hybrid` 模式会读取 `$SCENEEXPERT_MEMORY_DIR/ablation_4/indexes/` 下的 numpy index，并执行：

1. 按 `memory_type + stage` 做向量召回。
2. 用 `room_type`、required objects、failure `scope/is_deterministic` 做结构化过滤。
3. 用 embedding similarity、object overlap、stage/room match、memory quality、verified/deterministic signal 做 hybrid score 排序。

如果某个非空 memory bank 没有对应 index，当前默认策略是先自动构建缺失的 numpy index；只有 BGE-M3 本地模型、`FlagEmbedding` 依赖或写入目录不可用时才会停止。正式跑大量实验前仍建议手动执行一次 `scripts/build_memory_index.py`，这样可以把索引构建耗时从场景生成日志中剥离出来。

### PR-4 memory 消融配置

PR-4 后推荐用三组独立实验对比 memory 检索方式：

| 实验名 | 检索方式 | 运行前准备 |
| --- | --- | --- |
| `ablation_4a_qwen3_lexical_memory` | lexical token/alias overlap | 不需要 BGE-M3，不需要 index。 |
| `ablation_4b_qwen3_vector_memory` | BGE-M3 embedding + numpy 向量召回 | 需要安装 `requirements-memory.txt`；index 可自动构建，正式实验建议提前构建。 |
| `ablation_4c_qwen3_hybrid_memory` | structured filter + vector recall + hybrid score | 需要安装 `requirements-memory.txt`；index 可自动构建，推荐正式 memory 实验使用。 |

`ablation_4a/4b/4c` 会在各自的 experiment YAML 中显式设置 `retriever_type`。因此运行这些实验时，通常不需要在 `.env` 里手动改 `SCENEEXPERT_MEMORY_RETRIEVER_TYPE`。

如果已经有对应 memory JSONL，分别构建 4b/4c 的 numpy index：

```bash
# 4b: vector memory
python scripts/build_memory_index.py \
  --memory-dir "$SCENEEXPERT_MEMORY_DIR/ablation_4b" \
  --embedding-model-dir "$SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR" \
  --index-backend numpy \
  --device cpu

# 4c: hybrid memory
python scripts/build_memory_index.py \
  --memory-dir "$SCENEEXPERT_MEMORY_DIR/ablation_4c" \
  --embedding-model-dir "$SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR" \
  --index-backend numpy \
  --device cpu
```

运行：

```bash
bash scripts/run_experiment.sh ablation_4a_qwen3_lexical_memory
bash scripts/run_experiment.sh ablation_4b_qwen3_vector_memory
bash scripts/run_experiment.sh ablation_4c_qwen3_hybrid_memory
```

注意：4b/4c 默认 `index.require_ready=true` 且 `index.auto_build_missing=true`。也就是说，memory bank 里已有 JSONL 但没有 index 时，程序会先尝试自动构建；构建失败才会停止。这可以避免 ACP 任务因为忘记手动运行 `build_memory_index.py` 而在场景开始前直接退出。

## 10. 配置文件使用建议

推荐直接使用 `.env.example` 生成项目根目录 `.env`：

```bash
cp .env.example .env
vim .env
```

需要在当前终端手动检查变量时，先加载：

```bash
source .env
echo "$SCENEEXPERT_MODEL_ID"
```

不同运行场景可以维护多份 env 文件，例如：

```text
/share/configs/sceneexpert_qwen35.env
/share/configs/sceneexpert_qwen36.env
/share/configs/sceneexpert_lora.env
```

运行时指定：

```bash
SCENEEXPERT_ENV_FILE=/share/configs/sceneexpert_lora.env \
  bash scripts/run_experiment.sh ablation_5_qwen3_full
```

如果只想临时覆盖某一个值，可以在命令前加环境变量，不需要改 `.env`：

```bash
SCENEEXPERT_MAX_MODEL_LEN=65536 \
  bash scripts/run_experiment.sh ablation_4_qwen3_harness_memory
```

推荐分层：

- `.env`：服务器固定配置，如模型目录、数据目录、输出目录、vLLM 端口。
- `scripts/acp_sceneexpert.sh` 的 TODO 区：ACP 多卡作业配置，如实验名、GPU 数、`CUDA_VISIBLE_DEVICES`、上下文长度、CPU offload。
- Hydra override：实验差异，如 `experiment.pipeline.stop_stage=furniture`、`experiment.csv_path=...`。
- 临时环境变量：一次性资源调整，如 `SCENEEXPERT_MAX_MODEL_LEN=65536`。

## 11. 常见问题

### vLLM 启动失败

查看：

```bash
tail -f vllm_server.log
```

常见原因：

- 模型目录不完整。
- `SCENEEXPERT_MODEL_ID` 和 vLLM `--served-model-name` 不一致。
- vLLM 版本不支持 `--tool-call-parser qwen3_xml` 或 `--reasoning-parser qwen3`。
- 显存不足，降低 `SCENEEXPERT_MAX_MODEL_LEN` 或增加 tensor parallel GPU 数量。

如果日志里出现：

```text
World size (2) is larger than the number of available GPUs (1)
```

说明 `.env` 中的 `SCENEEXPERT_TENSOR_PARALLEL_SIZE` 大于当前 job 实际可见 GPU 数。根治方案二选一：

```bash
# 路线 A：当前只有 1 张 GPU，降低并行数和上下文长度
export SCENEEXPERT_TENSOR_PARALLEL_SIZE=1
export SCENEEXPERT_MAX_MODEL_LEN=65536

# 路线 B：保留 2 卡并行，向集群申请 2 张 GPU，并确认 CUDA_VISIBLE_DEVICES 里有 2 个设备
nvidia-smi
echo "$CUDA_VISIBLE_DEVICES"
```

Qwen3.5-35B-A3B 单卡运行时，`SCENEEXPERT_MAX_MODEL_LEN=262144` 很容易继续触发显存不足；建议先用 `32768` 或 `65536` 跑通流程。

如果日志里出现：

```text
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at ".../CUDACachingAllocator.cpp"
```

并且 traceback 停在 `FusedMoE` / `torch.empty` / `load_model` 附近，优先按显存容量问题处理。单卡 H100 80GB 跑未量化 Qwen3.5-35B-A3B 时，最直接的修复是开启 CPU offload：

```bash
export SCENEEXPERT_TENSOR_PARALLEL_SIZE=1
export SCENEEXPERT_MAX_MODEL_LEN=32768
export SCENEEXPERT_GPU_MEMORY_UTILIZATION=0.85
export SCENEEXPERT_VLLM_CPU_OFFLOAD_GB=20
```

如果仍失败，不建议继续微调小参数，直接换更稳的资源形态：

```bash
# 2 张或更多 80GB GPU
export SCENEEXPERT_TENSOR_PARALLEL_SIZE=2
export SCENEEXPERT_MAX_MODEL_LEN=65536

# 或者换成当前 vLLM 支持的量化模型，并设置对应量化方式
export SCENEEXPERT_VLLM_QUANTIZATION="fp8"
```

`nvidia-smi` 中的 `Insufficient Permissions` 不作为 SceneExpert 脚本阻断条件；当前脚本只提示并继续启动。

### vLLM engine core 启动超时

如果 `vllm_server.log` 最后出现：

```text
TimeoutError: Timed out waiting for engine core processes to start.
Waited 600s (configured by VLLM_ENGINE_READY_TIMEOUT_S).
```

这是 vLLM 内部 engine ready 超时，不是 OpenAI key 问题，也不是 SceneExpert 主流程问题。你的日志中模型已经完成了权重加载和 `torch.compile`，但 Qwen3.5-35B-A3B 在 2 卡 TP、AFS/FUSE 模型目录、首次编译场景下，engine core 初始化超过了 vLLM 默认 600 秒限制。

当前脚本已经把 ACP 默认值设为：

```bash
ACP_VLLM_WAIT_TIMEOUT_SECONDS=7200
ACP_VLLM_ENGINE_READY_TIMEOUT_S=7200
ACP_SAFETENSORS_LOAD_STRATEGY="prefetch"
```

其中 `ACP_VLLM_WAIT_TIMEOUT_SECONDS` 控制外层脚本等 `/health` 的时间；`ACP_VLLM_ENGINE_READY_TIMEOUT_S` 会被脚本导出为 vLLM 识别的 `VLLM_ENGINE_READY_TIMEOUT_S`，控制 vLLM API server 等 engine core 的内部时间。两者都要足够长，只改外层等待不够。

如果 ACP 日志只停在“vLLM PID=...，等待就绪...”附近，没有打印外层超时、退出码或 traceback，而 `vllm_server.log` 仍在写入编译 / warmup 日志，通常不是 Python 主程序主动退出，而是平台侧可能因为控制台长时间无输出结束任务。当前脚本已在等待循环中每 60 秒打印一次心跳；下一次任务应能看到：

```text
等待 vLLM 就绪: 60/7200s
vLLM 最新日志: ...
```

### DeepGEMM backend 缺失

如果 `vllm_server.log` 最后出现：

```text
RuntimeError: DeepGEMM backend is not available or outdated.
Please install or update the `deep_gemm` to a newer version to enable FP8 kernels.
```

这是 vLLM 在启动末尾执行 `deep_gemm_warmup` 时失败。你的日志中权重加载、`torch.compile`、initial profiling 都已经完成，失败点不是模型路径、不是 GPU 数量，也不是外层等待超时，而是当前离线环境没有兼容的 `deep_gemm` 后端。

当前脚本默认选择最稳的规避路线：禁用 DeepGEMM，并强制 vLLM 走 eager 模式。原因是当前 vLLM 0.22.x 日志里已经出现过 `DeepGEMM: use=0, moe_use=0, warmup=skip`，但 vLLM config 仍是 `enforce_eager=False`，随后继续进入 `torch.compile` / `kernel_warmup` / `deep_gemm_warmup` 并失败。因此只设置 DeepGEMM 环境变量不够，必须同时打开 `--enforce-eager`。

```bash
# ACP 脚本 TODO 区
ACP_VLLM_USE_DEEP_GEMM=0
ACP_VLLM_MOE_USE_DEEP_GEMM=0
ACP_VLLM_DEEP_GEMM_WARMUP="skip"
ACP_VLLM_ENFORCE_EAGER=1
```

`scripts/run_experiment.sh` 会把这些值转换为 vLLM 真实读取的环境变量：

```bash
VLLM_USE_DEEP_GEMM=0
VLLM_MOE_USE_DEEP_GEMM=0
VLLM_DEEP_GEMM_WARMUP=skip
```

并且会给 vLLM 启动命令追加：

```bash
--enforce-eager
```

下一次 ACP 任务提交后，先检查 `error.log` 是否打印：

```text
enforce eager: 1
```

再检查 `vllm_server.log` 的 engine config 是否变成：

```text
enforce_eager=True
```

如果仍然是 `enforce_eager=False`，说明 ACP 任务跑的不是当前仓库里的新脚本，或者提交前没有重新生成新的 job env 文件。

只有当集群镜像已经预装了与当前 vLLM/CUDA/PyTorch 匹配的 `deep_gemm` 时，才建议重新开启 DeepGEMM。

### vLLM worker 启动前显存不足

如果 `vllm_server.log` 里出现：

```text
ValueError: Free memory on device cuda:1 (...) on startup is less than desired GPU memory utilization
EngineCore failed to start
```

这和 DeepGEMM 的 `EngineCore failed to start` 不是同一个问题。根因是某张可见 GPU 在 vLLM 启动前已经有显存占用，例如日志里：

```text
GPU 状态:
0, NVIDIA H100 80GB HBM3, 81559 MiB, 0 MiB, 81080 MiB
1, NVIDIA H100 80GB HBM3, 81559 MiB, 37155 MiB, 43925 MiB
```

此时 `gpu_memory_utilization=0.90` 会要求每张 H100 大约 71GiB 可用显存，但 `cuda:1` 只剩约 43GiB，vLLM worker 会在初始化阶段直接退出。

优先修复：

```bash
# ACP 脚本 TODO 区保持空值，使用调度器分配的 GPU
ACP_CUDA_VISIBLE_DEVICES=""
```

并重新提交一个干净的 ACP 作业。如果仍然有占用，说明该 ACP 节点/分配的 GPU 不干净，需要换卡、换节点或清理残留进程。`scripts/run_experiment.sh` 默认开启 `SCENEEXPERT_GPU_PREFLIGHT_CHECK=1`，会在启动 vLLM 前提前拦截这类错误。

### Windows CRLF 导致 shell 解释失败

如果 ACP 的 `error.log` 很快出现：

```text
scripts/acp_sceneexpert.sh: line 11: $'\r': command not found
scripts/acp_sceneexpert.sh: line 12: set: pipefail
: invalid option name
```

说明脚本文件被 Windows 保存成了 CRLF 换行，Linux bash 会把行尾 `\r` 当成命令内容。项目已添加 `.gitattributes` 强制 `*.sh`、`*.py`、`*.yaml`、`*.md` 等文本文件使用 LF；所有 shell 入口也会在读取 `.env` 时自动去掉 CRLF。

如果集群上的工作副本已经被 CRLF 污染，可以在 Linux 节点执行一次：

```bash
find scripts -name '*.sh' -type f -print0 | xargs -0 sed -i 's/\r$//'
sed -i 's/\r$//' .env 2>/dev/null || true
```

然后重新提交 ACP。之后在 Windows 上修改代码时，不要用会强制 CRLF 的编辑器设置；提交前可检查：

```bash
grep -RIl $'\r' scripts --include='*.sh'
```

该命令没有输出才说明 shell 脚本是 LF。

### 找不到 HSSD、ArtVIP、materials

确认：

```bash
ls "$SCENEEXPERT_HSSD_DATA_DIR/hssd-models"
ls "$SCENEEXPERT_HSSD_DATA_DIR/preprocessed"
ls "$SCENEEXPERT_DATA_DIR/materials/embeddings"
ls "$SCENEEXPERT_DATA_DIR/artvip_sdf/embeddings"
```

本项目的最小快速复现可以只使用 HSSD 静态资产。若日志最后出现：

```text
Articulated source 'artvip' data path does not exist
RuntimeError: Articulated retriever initialization failed
```

说明当前数据目录没有 ArtVIP/PartNet articulated 资产，但配置仍尝试启动 articulated retrieval server。最快修复是保持：

```bash
export SCENEEXPERT_DISABLE_ARTICULATED=1
```

或者在 ACP 脚本 TODO 区保持：

```bash
ACP_DISABLE_ARTICULATED=1
```

`scripts/run_experiment.sh` 会自动追加以下 Hydra 覆盖，关闭四个 agent 的 articulated 策略：

```bash
furniture_agent.asset_manager.router.strategies.articulated.enabled=false
manipuland_agent.asset_manager.router.strategies.articulated.enabled=false
wall_agent.asset_manager.router.strategies.articulated.enabled=false
ceiling_agent.asset_manager.router.strategies.articulated.enabled=false
```

只有当你已经准备好 `artvip_sdf/embeddings` 或 `partnet_mobility_sdf/embeddings`，并希望使用可开合柜门、抽屉等 articulated 资产时，才把该开关改成 `0`。

若日志最后出现：

```text
Materials data path does not exist: .../materials
RuntimeError: Materials retriever initialization failed
```

说明当前数据目录没有 AmbientCG materials 资产或 embeddings，但配置仍尝试启动 materials retrieval server。最快修复是保持：

```bash
export SCENEEXPERT_DISABLE_MATERIALS=1
```

或者在 ACP 脚本 TODO 区保持：

```bash
ACP_DISABLE_MATERIALS=1
```

`scripts/run_experiment.sh` 会自动关闭 floor-plan 材料检索和 `thin_covering` 策略。只有当你已经准备好 `materials/` 和 `materials/embeddings/`，并希望启用材质语义检索或地毯、挂画、桌布等薄覆盖物资产时，才把该开关改成 `0`。

也可以在运行时显式覆盖：

```bash
bash scripts/run_experiment.sh ablation_2_qwen3_naive \
  furniture_agent.asset_manager.hssd.data_path="$SCENEEXPERT_HSSD_DATA_DIR/hssd-models" \
  furniture_agent.asset_manager.hssd.preprocessed_path="$SCENEEXPERT_HSSD_DATA_DIR/preprocessed" \
  experiment.materials_retrieval_server.data_path="$SCENEEXPERT_DATA_DIR/materials" \
  experiment.materials_retrieval_server.embeddings_path="$SCENEEXPERT_DATA_DIR/materials/embeddings"
```

如果 `git lfs install` 报 `git: 'lfs' is not a git command`，说明节点没有安装 Git LFS。按第 6 节的 HSSD 下载说明安装 `git-lfs`，或者改用 `huggingface-cli download hssd/hssd-models --repo-type dataset --local-dir "$SCENEEXPERT_HSSD_DATA_DIR/hssd-models"`。如果 `$SCENEEXPERT_HSSD_DATA_DIR` 在集群上不可写，请在数据构建机准备好后再挂载到集群。

### 不想使用外部图像生成 API

默认薄覆盖物材质生成可能配置了 OpenAI/Gemini fallback。无外网集群建议禁用 fallback：

```bash
bash scripts/run_experiment.sh ablation_4_qwen3_harness_memory \
  furniture_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false \
  manipuland_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false
```

### 端口冲突

```bash
export SCENEEXPERT_VLLM_PORT=18000
export OPENAI_BASE_URL="http://localhost:18000/v1"
bash scripts/run_experiment.sh ablation_2_qwen3_naive
```

### 已有 vLLM 服务，不想让脚本重复启动

```bash
export SCENEEXPERT_START_VLLM=0
export OPENAI_BASE_URL="http://localhost:8000/v1"
bash scripts/run_experiment.sh ablation_4_qwen3_harness_memory
```

### SceneExpert 没有产生 trace 或 memory

优先检查运行的 experiment 是否是 `ablation_3/4/5`。`ablation_2_qwen3_naive` 会禁用 SceneExpert，不会产生 SceneExpert trace/memory。`ablation_3` 只写 trace，不写 memory；`ablation_4/5` 才会读写 memory。

### 多房间场景变慢

启用 SceneExpert 时，代码会自动关闭 room-level parallelism，以保证 hook runner 和 memory 写入行为可控。多 prompt 批量实验可以用 CSV 加 `experiment.num_workers` 做 scene-level 并行；memory 模式下建议先保持 `experiment.num_workers=1`。

## 12. 最短命令清单

假设依赖、模型、数据都已经放到共享盘：

```bash
cd /path/to/SceneExpert
source .venv/bin/activate

cp .env.example .env
vim .env

bash scripts/run_experiment.sh ablation_4_qwen3_harness_memory
```

如果要切换模型：

```bash
SCENEEXPERT_ENV_FILE=/share/configs/sceneexpert_qwen36.env \
  bash scripts/run_experiment.sh ablation_4_qwen3_harness_memory
```
