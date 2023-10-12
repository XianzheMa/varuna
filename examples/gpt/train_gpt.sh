#! /bin/bash

kill $(lsof -t -i:29500)



python -m varuna.run_varuna --batch_size 288 --chunk_size 1 --gpus_per_node 1 \
	--machine_list machine_list.txt --trace_file ./a100_trace.json --env k8s \
	gpt_script.py --fp16 \
	--seq_length 2048 --embedding_dim 768 --num_heads 12 --num_layers 12 \
	--lr 0.00001 --save_every 10 --log_every 1 --num_iters 0
