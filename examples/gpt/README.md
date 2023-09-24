## troubleshooting
- If there are weird errors during model `.forward()` execution,
try to clear all tmp files with prefix `_tmp_` in `examples/gpt/` directory. They are cache storing structure information
about the model. If you change the model structure, you need to clear them.
