
import os, subprocess
from argparse import ArgumentParser, REMAINDER
import socket, time
import sys
import shlex
from .utils import HEARTBEAT_IP_ENV_VAR, HEARTBEAT_PORT_ENV_VAR, VARUNA_TEMP_FOLDER, MORPH_PORT_ENV_VAR

HEARTBEAT_PORT = 5000
MORPH_PORT = 4200

# TODO: readme/docs for launch process

launch_args_filename = "launch_args"
# TODO move this to args/utils
running_machines_list = "varuna_current_machines"

def check_morph_listeners(manager_ip):
    running = True
    # the morph server is completely developed. For our sake to only simulate a trace, we don't need it.
    for port in [HEARTBEAT_PORT]: #, MORPH_PORT]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((manager_ip, port))
            sock.sendall(bytes("is_running?", 'ascii'))
            response = str(sock.recv(1024), 'ascii')
            running = running and ("yes" in response)
    return running

def start_morph_listeners(available_machine_list, args):
    # TODO : need to ssh into manager ip and run these
    # morph server
    # cmd = f"python -m varuna.morph_server " \
    #       + f"{available_machine_list} {running_machines_list} {MORPH_PORT} "\
    #       + " > varuna_morph.out 2>varuna_morph.err &"
    # os.system(cmd)
    # TODO : check system errors

    # heartbeat server
    server_name = 'varuna.catch_all' if args.trace_file is None else 'varuna.catch_all_trace'
    cmd = f"python -m {server_name} " \
          + f"{running_machines_list} {HEARTBEAT_PORT} {args.trace_file} {args.env} {args.user_name} "\
          + " > varuna_catch.out 2>varuna_catch.err &"
    print(cmd)
    os.system(cmd)
    # wait for startup
    time.sleep(5)

def kill_morph_listeners():
    # os.system("pkill -f varuna.morph")
    cmd = f"pkill -f varuna.catch; kill $(lsof -t -i:{HEARTBEAT_PORT})"
    print(cmd)
    os.system(cmd)

def get_launch_cmd_format(args):
    launch_cmd = [sys.executable]
    launch_cmd.append(f" -u -m varuna.launcher" \
        +  f" --ngpus_per_server {args.gpus_per_node}  " \
        +  " --node_rank {} --nservers {} --master_addr {}"
        +  f"{' --nstages ' + str(args.nstages) if args.nstages is not None else ''} --batch_size {args.batch_size}" \
        +  f" --chunk_size {args.chunk_size} --code_dir {args.code_dir}")
    launch_cmd.append(args.training_script)
    launch_cmd.extend(args.training_script_args)
    launch_cmd = " ".join(launch_cmd)
    return launch_cmd

def get_env_vars(file):
    if not os.path.exists(file):
        return ""
    f =  open(file,"r")
    envstr = ""
    for line in f:
        envstr += line.strip("\n")
        return envstr

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Varuna training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")
    parser.add_argument("--machine_list", type=str, default="available_machines.out",
                            help = "path to a file with reachable IPs written line-wise." 
                            "These should be available to ssh into through the manager ")
    parser.add_argument("--gpus_per_node", type=int, default=4,
                            help = "number of GPUs per machine")
    parser.add_argument("--manager_ip", type=str, default=None,
                            help= "IP address for long-living manager, used for varuna morphing."
                                  "If not given, it defaults to the IP of the machine from which varuna is triggered.")
    parser.add_argument("--no_morphing", action="store_true",
                        help = "disable varuna's support for job morphing on a changing resource set.")
    parser.add_argument("--env_file", type=str, default="varuna.env",
                        help = "file with environment variables for varuna command")

    # launch worker args
    parser.add_argument("--nstages", type=int, default=None,
                        help="Depth of pipeline (number of stages)")
    parser.add_argument("--batch_size", default=None, type=int,
                        help="Total effective batch size for training")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="Micro-batch size per mini-batch")
    parser.add_argument("--code_dir", default=None, type=str,
                        help="Path on all machines of directory to run training in."
                        "Defaults to path from which launched.")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume a varuna run.")

    parser.add_argument('--trace_file', type=str, default=None, help='path to the trace file'
                                                                     'representing the resource changes over time')
    parser.add_argument('--user_name', type=str, default='', help='user name to ssh into worker nodes')
    parser.add_argument('--env', type=str, choices=['vm', 'k8s'], default='vm', help='environment to run varuna')
    parser.add_argument('--times', type=int, default=0, help='if set, denotes this is the n-th run due to resource changes.'
                                                                  'used as the suffix on log files.')
    parser.add_argument("training_script", type=str, default=None, nargs='?',
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    with open(args.machine_list, "r") as f:
        content = f.read()
        reachable_machines = content.split("\n")
        reachable_machines = [m for m in reachable_machines if len(m) > 0]  
        with open(running_machines_list, "w") as of:
            of.write(content)
    print(reachable_machines)  
    
    reachable_count = len(reachable_machines)
    if reachable_count == 0:
        print("Empty machine list, nothing to run!")
        exit()

    if any([arg is None for arg in [args.batch_size, args.chunk_size, args.training_script]]):
        assert args.resume, "Training script, batch size, and micro-batch size required!"

    if args.code_dir is None:
        args.code_dir = os.getcwd()

    if args.manager_ip is None:
        args.manager_ip = socket.gethostbyname(socket.gethostname())
    print(f'manager ip is {args.manager_ip}')

    if not args.no_morphing:
        if not args.resume:
            kill_morph_listeners()
            start_morph_listeners(args.machine_list, args)
        assert check_morph_listeners('127.0.0.1'), "Listeners not in place for morphing!"

    if not os.path.exists(VARUNA_TEMP_FOLDER):
        os.makedirs(VARUNA_TEMP_FOLDER)
    arg_file = os.path.join(VARUNA_TEMP_FOLDER, launch_args_filename)
    if args.resume:
        assert os.path.exists(arg_file), "Args file not found for resumed run!"
        with open(arg_file, "r") as f:
            launch_cmd_format = f.read()
    else:
        launch_cmd_format = get_launch_cmd_format(args)
        with open(arg_file, "w") as f:
            f.write(launch_cmd_format)

    master_addr = reachable_machines[0]
    if args.env == 'k8s':
        # resolve the address
        cmd = f"kubectl get pod -o wide | grep {master_addr} | awk '{{print $6}}'"
        print(cmd)
        master_addr = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        print(f'resolved address is {master_addr}')

    os.makedirs("ssh_logs", exist_ok=True)
    current_env = os.environ.copy()
    current_env[HEARTBEAT_IP_ENV_VAR] = str(args.manager_ip)
    current_env[HEARTBEAT_PORT_ENV_VAR] = str(HEARTBEAT_PORT)
    current_env[MORPH_PORT_ENV_VAR] = str(MORPH_PORT)
   # current_env["PATH"] = "PATH=\"/home/varuna/anaconda3/bin:$PATH\""

    processes = []

    # FIXME: a hack to determine if this is a profiling run or a training run
    if '--profiling' in args.training_script_args:
        log_file_suffix = f'profiling_{args.times}'
    else:
        log_file_suffix = f'training_{args.times}'

    for i,machine in enumerate(reachable_machines):
        launch_cmd = launch_cmd_format.format(i, reachable_count, master_addr)
        # the log_file_suffix is to distinguish between profiling and training runs,
        # and distinguish different (re)-runs of training runs. Otherwise the logs are overwritten.
        out_file = open(f"ssh_logs/ssh_out_{i}_{log_file_suffix}", "w")
        err_file = open(f"ssh_logs/ssh_err_{i}_{log_file_suffix}", "w")

        if args.env == 'k8s':
            # FIXME: do not support args.env_file
            cmd = f"kubectl exec {machine} -- /bin/bash -c "
            # FIXME: hack to add times
            cmd = cmd + f"\'{HEARTBEAT_IP_ENV_VAR}={socket.gethostname()} {MORPH_PORT_ENV_VAR}={MORPH_PORT} {HEARTBEAT_PORT_ENV_VAR}={HEARTBEAT_PORT} {launch_cmd} --times {args.times} \'"
            print(cmd)
            cmd = shlex.split(cmd)
        else:
            if machine == "127.0.0.1":
                cmd = launch_cmd.split(" ")
                cmd = [x_ for x_ in cmd if x_ != ""]
                print("launch cmd is ", cmd)
            else:
                cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
                cmd.append((args.user_name + '@' if len(args.user_name) > 0 else '') + machine)
                cmd.append(f"echo \"{launch_cmd}\" > launch_varuna.sh; ")
                cmd.append(f"{HEARTBEAT_IP_ENV_VAR}={socket.gethostname()}")
                cmd.append(f"{MORPH_PORT_ENV_VAR}={MORPH_PORT} {HEARTBEAT_PORT_ENV_VAR}={HEARTBEAT_PORT}")
                cmd.append(get_env_vars(args.env_file))
                cmd.append("bash launch_varuna.sh")
                print(" ".join(cmd ))
       
        process = subprocess.Popen(cmd, env=current_env, stdout=out_file, stderr=err_file)
        processes.append(process)

    # FIXME: a hack to wait for profiling processes to finish
    if '--profiling' in args.training_script_args:
        # wait for all processes
        try:
            for process in processes:
                process.wait()
                print("Process done with return code", process.returncode)
                if process.returncode != 0:
                    for p in processes:
                        p.kill()
        except Exception as e:
            print("run_varuna subprocesses quit with error:", e)
