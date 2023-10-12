## Troubleshooting
- If there are weird errors during model `.forward()` execution,
try to clear all tmp files with prefix `_tmp_` in `examples/gpt/` directory. They are cache storing structure information
about the model. If you change the model structure, you need to clear them.

## Understanding
Upon executing `run_varuna.py` on master node, it spins up two servers locally:

### `catch_all`

The rank 0 worker node periodically sends a heartbeat to this server, reporting the current iteration.
If after a certain amount of time, the current iteration is not updated on the server side, the server will kill all workers
and trigger some restart logic (it is not completely implemented in the original repo).

#### changes to make
We will not re-trigger restart logic based on iteration, but given a trace file, we will trigger restart logic
once resource changes according to the trace file. We consolidated all changes to another file `trace_catch_all.py`.


### `morph_server`

`morph_server` is responsible for many tasks but most of them are broken.
For our purposes we do not use it.

## Miscellaneous

1. [kill_all.sh](varuna/kill_all.sh) is a script to kill all processes on all nodes. It is used by `trace_catch_all.py` / 
`catch_all.py` to kill all processes when the program exits. Originally what's killed is `varuna.launcher` but I found
the processes spawned by `varuna_launcher` (the actual training script, in our case [gpt_script.py](examples/gpt/gpt_script.py))
are not killed. So I changed it to kill the actual training script by name `gpt_script`.

    If you create a new training script, you need to change the name in `kill_all.sh` accordingly.

2. The original varuna is meant to run directly on a set of VMs. I made it also possible
to run on a kubernetes cluster, with each worker node / the manager as a pod. In `examples/gpt/run/` there are sample k8s resource
yaml files. To run on k8s, you need to set `run_varuna.py`'s `--env` arg as `k8s`, which defaults to `vm`.

