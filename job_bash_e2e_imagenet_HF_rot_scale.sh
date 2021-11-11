#PBS -N no_aug_imagenet_HF_rot_scale
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4:gpus=1:ubuntu2004:nvidiaGTX1080Ti,mem=8gb,walltime=24:00:00
#PBS -j oe
#PBS -o /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/outputs_lincls_aug/


homePath='/misc/student/alzouabk/miniconda3'
source $homePath/bin/activate Thesis_CT_scans

echo "pid, gpu_utilization [%], mem_utilization [%], max_memory_usage [MiB], time [ms]"
nvidia-smi --query-accounted-apps="pid,gpu_util,mem_util,max_memory_usage,time" --format=csv | tail -n1

echo 'Training Should start'
python3 /misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/main_lincls.py \
  --exp 'no_aug_imagenet_HF_rot_scale' \
  --output_dir='/misc/student/alzouabk/Thesis/self_supervised_pretraining/moco/outputs_lincls_aug/' \
  --epochs 40 \
  --e2e \
  --optimizer adam \
  --print-freq 1 \
  --arch resnet18 \
  --from_imagenet \
  --lr 0.001 \
  --batch-size 16 \
  --RHF-p=0.5 \
  --affine-rot=45 \
  --affine-scale 0.5 1.5 \
  --affine-p=1.0 \
  --dist-url 'tcp://localhost:10007' --multiprocessing-distributed --world-size 1 --rank 0 \
  --workers 8 \
