#!/bin/sh
#PBS -q rt_HF
#PBS -N qwen3-8b-sft
#PBS -l select=8
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -koed
#PBS -V
#PBS -o outputs/qwen3-8b-sft/

set -e
cleanup() {
  echo "Cleaning up Ray processes..."
  ray stop
  kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT SIGTERM

cd $PBS_O_WORKDIR

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles

module load cuda/12.9.1
module load cudnn/9.10.2
module load nccl/2.27.5-cuda12.9
module load hpcx/2.20

JOB_ID=$(echo $PBS_JOBID | cut -d. -f1)
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"
export NUM_GPU_PER_NODE=8
NODE_TYPE="h200"

NODEFILE=$PBS_NODEFILE
NODE_COUNT=$(sort -u $NODEFILE | wc -l)
NUM_NODES=$NODE_COUNT
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
sort -u "$PBS_NODEFILE" | while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done >"$HOSTFILE_NAME"

# Training Settings
TENSOR_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=2
EXPERT_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
SEQUENCE_PARALLEL=False
# if [[ $TENSOR_PARALLEL_SIZE -gt 1 ]]; then
#   SEQUENCE_PARALLEL=True
# fi

LR=1.5E-5
MIN_LR=1.5E-6
GLOBAL_BATCH_SIZE=64
MICRO_BACH_SIZE=1

TRAIN_ITERATION=7800
WARMUP_ITERATION=780
SEQ_LENGTH=32768

TOKENIZER_CHAT_TEMPLATE="/groups/gch51639/fujii/checkpoints/megatron-to-hf/Qwen3-Swallow-8B-v0.1-SFT/swallow-reasoning/exp18/iteration_0007800/chat_template.jinja"

DATASET_PATH="/groups/gch51639/fujii/datasets/raw/instruct/swallow/Qwen3-Swallow-SFT/exp15/train.jsonl"

# Checkpoint Settings
CHECKPOINT_DIR="tokyotech-llm/Qwen3-8B-Reasoning-exp6-LR1.5E-5-iter0012500"
CHECKPOINT_SAVE_DIR="/groups/gch51639/fujii/checkpoints/nemo-rl/sft/Qwen3-Swallow-8B-Reasoning/exp17/NODE_${NUM_NODES}/TP${TENSOR_PARALLEL_SIZE}_CP${CONTEXT_PARALLEL_SIZE}/LR_${LR}_MIN_LR_${MIN_LR}/"
mkdir -p $CHECKPOINT_SAVE_DIR

# Environment variable
export WANDB_ENTITY="prj-jalm"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export TORCH_CUDA_ARCH_LIST="9.0"
export UV_EXTRA_INDEX_URL="https://pypi.nvidia.com"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Ray Setup
mapfile -t NODES < <(sort -u "$PBS_NODEFILE")
HEAD_NODE=${NODES[0]}
WORKER_NODES=("${NODES[@]:1}")

export CONTROL_IFNAME="bond0"
export MASTER_ADDR=$(ip -4 addr show $CONTROL_IFNAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)
export HEAD_IP=$MASTER_ADDR
PORT=6379
echo "Head Node: $HEAD_NODE ($HEAD_IP)"
echo "Worker Nodes: ${WORKER_NODES[*]}"

# for DEBUG
export NCCL_DEBUG="INFO"
# 2. PyTorch Distributed / Gloo (制御通信)
# ここは bond0 を使います
export GLOO_SOCKET_IFNAME=$CONTROL_IFNAME

# 3. NCCL (GPU間通信 - データ転送)
# 重要: "^ibn" という正規表現を使って、ibn1～ibn8 すべてを使わせます。
# これにより 8つのレールを束ねて最大帯域を出します。
# export NCCL_SOCKET_IFNAME="^ibn"
export NCCL_SOCKET_IFNAME="bond0"
export NCCL_IB_HCA="^mlx5_ibn"  # -x で渡すの忘れずに
export NCCL_CROSS_NIC=1
export NCCL_ALGO=RING
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CACHE_ROOT="/groups/gch51639/fujii/cache"
mkdir -p $CACHE_ROOT/torch_extensions
mkdir -p $CACHE_ROOT/nvte_cache
mkdir -p $CACHE_ROOT/triton_cache
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export NVTE_CACHE_PATH="$CACHE_ROOT/nvte_cache"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton_cache"
export NVTE_FRAMEWORK_LOGGING=1

# Ray starting Head Node
echo "Starting Ray Head on $HEAD_NODE..."
ray start --head \
  --node-ip-address=$HEAD_IP \
  --port=$PORT \
  --num-gpus=$NUM_GPU_PER_NODE \
  --disable-usage-stats &

sleep 10  # waiting for head node launch

# Ray starting Worker Nodes
echo "Starting Ray Workers via mpirun..."
WORKER_CMD="
if [ \"\$(hostname)\" != \"$HEAD_NODE\" ]; then
  export PATH=$PATH;
  ray start --address=$HEAD_IP:$PORT --num-gpus=$NUM_GPU_PER_NODE --disable-usage-stats --block;
else
  echo \"I am Head ($HEAD_NODE), skipping worker start.\";
fi
"
mpirun -np ${#NODES[@]} \
  --map-by ppr:1:node \
  -x PATH \
  -x NCCL_SOCKET_IFNAME \
  -x GLOO_SOCKET_IFNAME \
  -x NCCL_IB_HCA \
  -x NCCL_CROSS_NIC \
  -x NCCL_ALGO \
  -x NCCL_DEBUG \
  bash -c "$WORKER_CMD" &

sleep 15  # waiting for worker nodes

# Job Launch
export RAY_ADDRESS="$HEAD_IP:$PORT"
echo "Running training script with RAY_ADDRESS=$RAY_ADDRESS"

uv run python examples/run_sft.py \
  --config examples/configs/sft.yaml \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${NUM_GPU_PER_NODE} \
  checkpointing.checkpoint_dir=${CHECKPOINT_SAVE_DIR} \
  checkpointing.save_period=500 \
  policy.model_name=${CHECKPOINT_DIR} \
  policy.tokenizer.chat_template=${TOKENIZER_CHAT_TEMPLATE} \
  policy.optimizer.kwargs.eps=1.0E-8 \
  policy.optimizer.kwargs.lr=${LR} \
  policy.dtensor_cfg.enabled=false \
  policy.megatron_cfg.enabled=true \
  policy.sequence_packing.enabled=true \
  policy.megatron_cfg.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE} \
  policy.megatron_cfg.expert_model_parallel_size=${EXPERT_PARALLEL_SIZE} \
  policy.megatron_cfg.pipeline_model_parallel_size=${PIPELINE_PARALLEL_SIZE} \
  policy.megatron_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE} \
  policy.megatron_cfg.sequence_parallel=${SEQUENCE_PARALLEL} \
  policy.megatron_cfg.optimizer.lr=${LR} \
  policy.megatron_cfg.optimizer.min_lr=${MIN_LR} \
  policy.megatron_cfg.optimizer.weight_decay=0.1 \
  policy.megatron_cfg.optimizer.bf16=true \
  policy.megatron_cfg.optimizer.adam_beta1=0.9 \
  policy.megatron_cfg.optimizer.adam_beta2=0.95 \
  policy.megatron_cfg.optimizer.adam_eps=1E-8 \
  policy.megatron_cfg.optimizer.use_precision_aware_optimizer=false \
  policy.megatron_cfg.scheduler.lr_decay_style="" \
  policy.megatron_cfg.scheduler.lr_decay_iters=${TRAIN_ITERATION} \
  policy.megatron_cfg.scheduler.lr_warmup_iters=${WARMUP_ITERATION} \
  policy.megatron_cfg.distributed_data_parallel_config.grad_reduce_in_fp32=true \
  policy.max_total_sequence_length=${SEQ_LENGTH} \
  policy.train_global_batch_size=${GLOBAL_BATCH_SIZE} \
  policy.train_micro_batch_size=${MICRO_BACH_SIZE} \
  sft.val_global_batch_size=${GLOBAL_BATCH_SIZE} \
  sft.val_micro_batch_size=${MICRO_BACH_SIZE} \
  sft.max_num_steps=${TRAIN_ITERATION} \
  sft.val_period=500 \
  sft.val_at_start=false \
  data.add_bos=false \
  data.add_eos=true \
  data.dataset_name="JSONLDataset" \
  +data.train_data_path=$DATASET_PATH \
  +data.val_data_path=$DATASET_PATH \
  +data.conversation_key="conversation" \
  logger.log_dir=${CHECKPOINT_SAVE_DIR}/logs \
  logger.monitor_gpus=false \
  logger.wandb_enabled=true \
  logger.wandb.project="nemo-rl" \
  logger.wandb.name="Qwen3-Swallow-8B-SFT-exp17-LR_${LR}_MIN_LR_${MIN_LR}"

# Clean
ray stop
kill $(jobs -p)
