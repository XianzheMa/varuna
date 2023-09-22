#! /bin/bash

kill $(lsof -t -i:29500)
WANDB_API_KEY=# enter your wandb api key here
wandb login $WANDB_API_KEY
export HF_DATASETS_CACHE=# enter your dataset cache path here
python -m varuna.run_varuna --nstages 2 --batch_size 288 --chunk_size 4 --gpus_per_node 4 \
	--machine_list machine_list.txt --no_morphing gpt_script.py --fp16 \
	--seq_length 2048 --embedding_dim 768 --num_heads 12 --num_layers 12 \
	--lr 0.00001
#tail -F ssh_logs/ssh_out_0
