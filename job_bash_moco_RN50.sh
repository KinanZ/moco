#PBS -N moco_RN50_0001
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4:gpus=2:nvidiaMin12GB,mem=16gb,walltime=24:00:00
#PBS -j oe
#PBS -q student
#PBS -o /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/moco_curves_2/


homePath='/misc/student/alzouabk/miniconda3'
source $homePath/bin/activate Thesis_CT_scans

echo "pid, gpu_utilization [%], mem_utilization [%], max_memory_usage [MiB], time [ms]"
nvidia-smi --query-accounted-apps="pid,gpu_util,mem_util,max_memory_usage,time" --format=csv | tail -n1

echo 'Training Should start'
python3 /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/main_moco.py \
  --exp 'RN50_0001' \
  --epochs 100 \
  --print-freq 10 \
  --arch resnet50 \
  --moco-dim 128 \
  --stack_pre_post True \
  --lr 0.0001 \
  --batch-size 48 \
  --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 \
  --workers 8 \

