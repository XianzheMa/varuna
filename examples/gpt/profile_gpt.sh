#! /bin/bash

python -m varuna.run_varuna --gpus_per_node 1 \
	--batch_size 288 --nstages 1 --chunk_size 1 --env k8s \
	--machine_list machine_list.txt --no_morphing gpt_script.py --fp16 \
	--seq_length 2048 --embedding_dim 768 --num_heads 12 --num_layers 12 \
	--lr 0.00001 --profiling
