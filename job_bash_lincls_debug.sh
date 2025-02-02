#PBS -N imagenet_lincl_old_2
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=2:nvidiaMin11GB,mem=16gb,walltime=24:00:00
#PBS -j oe
#PBS -q student
#PBS -o /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/outputs_lincls/


homePath='/misc/student/alzouabk/miniconda3'
source $homePath/bin/activate Thesis_CT_scans

echo "pid, gpu_utilization [%], mem_utilization [%], max_memory_usage [MiB], time [ms]"
nvidia-smi --query-accounted-apps="pid,gpu_util,mem_util,max_memory_usage,time" --format=csv | tail -n1

echo 'Training Should start'
python3 /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/main_lincls.py \
  --exp 'imagenet_lincl_old_2' \
  --epochs 40 \
  --print-freq 1 \
  --arch resnet50 \
  --lr 0.375 \
  --batch-size 48 \
  --pretrained /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/outputs/3c_v2_bestAug_2_128/checkpoint_0099.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --workers 8 \
