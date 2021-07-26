#PBS -N moco_1c_v1
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=2:nvidiaMin11GB,mem=16gb,walltime=24:00:00
#PBS -j oe
#PBS -q student
#PBS -o /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/outputs/


homePath='/misc/student/alzouabk/miniconda3'
source $homePath/bin/activate Thesis_CT_scans

echo "pid, gpu_utilization [%], mem_utilization [%], max_memory_usage [MiB], time [ms]"
nvidia-smi --query-accounted-apps="pid,gpu_util,mem_util,max_memory_usage,time" --format=csv | tail -n1

echo 'Training Should start'
python3 /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/main_moco.py \
  --exp '1c_v1_2' \
  --epochs 200 \
  --print-freq 10 \
  --arch resnet50 \
  --channel_num 1 \
  --lr 0.03 \
  --batch-size 64 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --workers 4 \

