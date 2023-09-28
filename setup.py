from setuptools import find_packages, setup
import glob
import subprocess, os


s = setup(
    name="varuna",
    version="0.0.1",
    author="MSR India",
    author_email="muthian@microsoft.com",
    description="Pipeline parallel training for PyTorch",
    keywords='deep learning microsoft research pipelining',
    packages=['varuna']
)

# the original script installed the binaries in the repository source path
# however they should be installed in the installed path
# e.g. the path containing "lib/python<version>/site-packages/..."
# this change is to fix that
installed_path = s.command_obj['install'].install_lib
target_this_dir = glob.glob(os.path.join(installed_path, 'varuna*.egg'))[0]
target_varuna_path = os.path.join(target_this_dir, 'varuna')
genschedule_path = os.path.join(target_varuna_path, "genschedule")

this_dir = os.path.dirname(os.path.abspath(__file__))
varuna_dir = os.path.join(this_dir, "varuna")


cmd = ["g++", "-std=c++11", "generate_schedule.cc", "-o", f"{genschedule_path}"]
print(f'Installing genschedule at {genschedule_path}', flush=True)
subprocess.run(cmd, cwd=varuna_dir, check=True)

target_kill_all_path = os.path.join(target_varuna_path, "kill_all.sh")
subprocess.run(["cp", "kill_all.sh", target_kill_all_path], cwd=varuna_dir, check=True)
subprocess.run(["chmod", "+x", target_kill_all_path], check=True)

tools_dir = os.path.join(this_dir, "tools", "simulator")
target_simulate_varuna_dir = os.path.join(target_this_dir, "tools", "simulator")
os.makedirs(target_simulate_varuna_dir, exist_ok=True)
simulate_varuna_path = os.path.join(target_simulate_varuna_dir, "simulate-varuna")
cmd = ["g++","-std=c++11", "simulate-varuna-main.cc", "generate_schedule.cc", "simulate-varuna.cc", "-o", f"{simulate_varuna_path}"]
print(f'Installing simulate-varuna at {simulate_varuna_path}', flush=True)
subprocess.run(cmd, cwd=tools_dir, check=True)
