export CUDA_VISIBLE_DEVICES="0, 1, 2"
export PYTHONPATH="."

python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=3 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg /lustre/scratch/client/vinai/users/trangpvh1/repo/vinai_prj/VMamba_Global_local_278/classification/configs/vssmab/vmambav0_tiny_224_a0.yaml --batch-size 128 --data-path /lustre/scratch/client/vinai/users/trangpvh1/imagenet --output /lustre/scratch/client/vinai/users/trangpvh1/repo/vinai_prj/VMamba_Global_local_278/classification --resume /lustre/scratch/client/vinai/users/trangpvh1/repo/vinai_prj/VMamba_Global_local_278/classification/vssm_tiny_v0/20240827180640/ckpt_epoch_3.pth 
