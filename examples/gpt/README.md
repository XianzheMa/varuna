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
