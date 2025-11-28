#!/bin/sh
#PBS -q rt_HF
#PBS -N qwen3-1.7b-sft
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -m n
#PBS -koed
#PBS -V
#PBS -o outputs/qwen3-1.7b-sft/

cd $PBS_O_WORKDIR

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles
module load cuda/12.9.1
module load hpcx/2.20

source .venv/bin/activate

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
TENSOR_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
SEQUENCE_PARALLEL=False
if [[ $TENSOR_PARALLEL_SIZE -gt 1 ]]; then
  SEQUENCE_PARALLEL=True
fi

LR=1.0E-5
MIN_LR=1.0E-6
GLOBAL_BATCH_SIZE=64
MICRO_BACH_SIZE=1

TRAIN_ITERATION=2500
SEQ_LENGTH=8192

# Checkpoint Settings
CHECKPOINT_DIR=
CHECKPOINT_SAVE_DIR="/groups/gch51639/fujii/checkpoints/nemo-rl/Llama-3.2-1B/LR_${LR}_MIN_LR_${MIN_LR}/"

# Ray Setup
mapfile -t NODES < <(sort -u "$PBS_NODEFILE")
HEAD_NODE=${NODES[0]}
WORKER_NODES=("${NODES[@]:1}")
HEAD_IP=$(hostname -i | awk '{print $1}')
PORT=6379
echo "Head Node: $HEAD_NODE ($HEAD_IP)"
echo "Worker Nodes: ${WORKER_NODES[*]}"

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
mpirun -np ${#NODES[@]} --map-by ppr:1:node -x PATH bash -c "$WORKER_CMD" &

sleep 15  # waiting for worker nodes

# Job Launch
export RAY_ADDRESS="$HEAD_IP:$PORT"
echo "Running training script with RAY_ADDRESS=$RAY_ADDRESS"

export WANDB_ENTITY="okoge"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

uv run ./examples/run_sft.py \
  --config examples/configs/sft.yaml \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${NUM_GPUS} \
  checkpointing.checkpoint_dir=${CHECKPOINT_SAVE_DIR} \
  policy.optimizer.kwargs.eps=1.0E-8 \
  policy.optimizer.kwargs.lr=${LR} \
  policy.dtensor_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE} \
  policy.dtensor_cfg.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
  policy.dtensor_cfg.sequence_parallel=${SEQUENCE_PARALLEL} \
  policy.max_total_sequence_length=${SEQ_LENGTH} \
  policy.train_global_batch_size=${GLOBAL_BATCH_SIZE} \
  policy.train_micro_batch_size=${MICRO_BACH_SIZE} \
  sft.val_global_batch_size=${GLOBAL_BATCH_SIZE} \
  sft.val_micro_batch_size=${MICRO_BACH_SIZE} \
  sft.max_num_steps=${TRAIN_ITERATION} \
  logger.wandb_enabled=True \
  logger.wandb.project="nemo-rl" \
  logger.wandb.name="sft-llama8b"

# Clean
ray stop
kill $(jobs -p)
