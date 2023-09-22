#! /bin/bash

python -m varuna.run_varuna --nstages 2 --batch_size 128 --chunk_size 4 --gpus_per_node 4 \
	--machine_list machine_list.txt --no_morphing test_script.py