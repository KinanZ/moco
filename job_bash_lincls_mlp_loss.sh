#PBS -N moco_lincl_mlp_RN18_001_loss
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4:gpus=2:nvidiaMin12GB,mem=16gb,walltime=24:00:00
#PBS -j oe
#PBS -q student
#PBS -o /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/outputs_lincls_2/


homePath='/misc/student/alzouabk/miniconda3'
source $homePath/bin/activate Thesis_CT_scans

echo "pid, gpu_utilization [%], mem_utilization [%], max_memory_usage [MiB], time [ms]"
nvidia-smi --query-accounted-apps="pid,gpu_util,mem_util,max_memory_usage,time" --format=csv | tail -n1

echo 'Training Should start'
python3 /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/main_lincls.py \
  --exp 'moco_lincl_mlp_RN18_001_loss' \
  --epochs 40 \
  --print-freq 1 \
  --optimizer adam \
  --arch resnet18 \
  --lr 0.001 \
  --batch-size 48 \
  --pretrained /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/outputs_2/mlp_RN18_001_loss/best_model_loss.pth.tar \
  --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0 \
  --workers 8 \

