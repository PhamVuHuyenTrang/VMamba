export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="."

python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg your_config_path --batch-size 32 --data-path your_data_path --output your_output_path > your_output_file_name.txt
