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
varuna_egg_path = glob.glob(os.path.join(installed_path, 'varuna*.egg'))[0]
target_binary_path = os.path.join(varuna_egg_path, 'varuna')

this_dir = os.path.dirname(os.path.abspath(__file__))
varuna_dir = os.path.join(this_dir, "varuna")
genschedule_path = os.path.join(target_binary_path, "genschedule")
simulate_varuna_path = os.path.join(target_binary_path, "simulate-varuna")


cmd = ["g++", "-std=c++11", "generate_schedule.cc", "-o", f"{genschedule_path}"]
subprocess.run(cmd, cwd=varuna_dir, check=True)
tools_dir = os.path.join(this_dir, "tools", "simulator")
cmd = ["g++","-std=c++11", "simulate-varuna-main.cc", "generate_schedule.cc", "simulate-varuna.cc", "-o", f"{simulate_varuna_path}"]
subprocess.run(cmd, cwd=tools_dir, check=True)
