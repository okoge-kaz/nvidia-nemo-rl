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

source .venv/bin/activate

source /etc/profile.d/modules.sh
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
