#! /bin/bash

kill $(lsof -t -i:29500)
#WANDB_API_KEY= # enter your wandb api key here
GPUS_PER_NODE=4
export HF_DATASETS_CACHE=/scratch/xianma/.cache/huggingface/datasets/ # enter your dataset cache path here

#wandb login $WANDB_API_KEY
python -m varuna.run_varuna --batch_size 288 --no_morphing --chunk_size 4 --gpus_per_node $GPUS_PER_NODE \
	--machine_list machine_list.txt gpt_script.py --fp16 \
	--seq_length 2048 --embedding_dim 768 --num_heads 12 --num_layers 12 \
	--lr 0.00001
#tail -F ssh_logs/ssh_out_0
