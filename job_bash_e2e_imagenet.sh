#PBS -N imagenet_e2e_resnet18_256
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4:gpus=1:nvidiaMin12GB,mem=16gb,walltime=24:00:00
#PBS -j oe
#PBS -q student
#PBS -o /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/outputs_lincls/


homePath='/misc/student/alzouabk/miniconda3'
source $homePath/bin/activate Thesis_CT_scans

echo "pid, gpu_utilization [%], mem_utilization [%], max_memory_usage [MiB], time [ms]"
nvidia-smi --query-accounted-apps="pid,gpu_util,mem_util,max_memory_usage,time" --format=csv | tail -n1

echo 'Training Should start'
CUDA_LAUNCH_BLOCKING=1 python3 /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/main_lincls.py \
  --exp 'imagenet_e2e_resnet18_256' \
  --epochs 40 \
  --e2e \
  --optimizer adam \
  --print-freq 1 \
  --arch resnet18 \
  --lr 0.001 \
  --batch-size 48 \
  --from_imagenet \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --workers 8 \
