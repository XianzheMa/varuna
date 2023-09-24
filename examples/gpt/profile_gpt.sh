#! /bin/bash

GPUS_PER_NODE=4
export HF_DATASETS_CACHE=# enter your dataset cache path here


python -m varuna.run_varuna --gpus_per_node $GPUS_PER_NODE \
	--batch_size 288 --nstages 1 --chunk_size 1 \
	--machine_list machine_list.txt --no_morphing gpt_script.py --fp16 \
	--seq_length 2048 --embedding_dim 768 --num_heads 12 --num_layers 12 \
	--lr 0.00001 --profiling
tail -F ./ssh_logs/ssh_out_0
