# a single-threaded loop that reads a trace file, and then starts/stops varuna accordingly
import json
import socket
import threading
import socketserver
import time
from datetime import datetime
import os
import sys
import subprocess
from threading import Thread
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s', stream=sys.stdout)
class Handler(socketserver.BaseRequestHandler):

    scripts_folder = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def kill_all(running_machines_list, user_name, env):
        logging.info("killing all")
        sh = os.path.join(Handler.scripts_folder, "kill_all.sh")
        p = None
        try:
            cmds = ['bash', sh, running_machines_list, env]
            if user_name != '':
                cmds.append(user_name)
            print(' '.join(cmds))
            p = subprocess.call(cmds, timeout=120)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logging.info(f"kill errored/timed out: {e}" )
            if p is not None:
                p.kill()

    @staticmethod
    def start_remote(running_machines_list, user_name, env, times):
        cmd = "python -m varuna.run_varuna --resume " + \
              f"--machine_list {running_machines_list} {'--user_name ' + user_name if len(user_name) > 0 else ''} " + \
              f"--env {env} --times {times}"
        logging.info(f"restart cmd is {cmd}")
        os.system(cmd)

    def handle(self):
        data = str(self.request.recv(1024), 'ascii')
        logging.info("{} got something from {}: {}".format(datetime.now(), self.client_address, data))

        if 'is_running?' in data:
            response = bytes("yes", 'ascii')
            self.request.sendall(response)
        else:
            logging.info(f"got unknown message: {data}")


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


def react_to_trace(trace_file, running_machines_list, user_name, env):

    with open(running_machines_list, 'r') as f:
        available_machines = [
            line.strip() for line in f.readlines() if line.strip() != ""
        ]

    with open(trace_file, 'r') as f:
        trace = json.load(f)

    initial_timestamp = datetime.now().timestamp()
    # we skip the 0-th entry because it is the initial state
    next_trace_change_id = 1

    while next_trace_change_id < len(trace):
        current_timestamp = datetime.now().timestamp()
        if current_timestamp - initial_timestamp < trace[next_trace_change_id]['Time']:
            logging.info('wait')
            time.sleep(3)
            continue

        num_training_nodes = trace[next_trace_change_id]['GPUs']
        logging.info(f"At {next_trace_change_id} Update to {num_training_nodes} nodes")
        training_machines = available_machines[:num_training_nodes]
        next_trace_change_id += 1
        Handler.kill_all(running_machines_list, user_name, env)
        # update running_machines_list
        with open(running_machines_list, 'w') as f:
            f.write("\n".join(training_machines))

        # otherwise it is the end of the trace so we don't need to restart
        if next_trace_change_id < len(trace):
            Handler.start_remote(running_machines_list, user_name, env, next_trace_change_id - 1)

    logging.info("trace ended")
    exit(0)

if __name__ == "__main__":
    running_machines_list = sys.argv[1]
    HOST = '0.0.0.0'
    PORT = int(sys.argv[2])
    trace_file = sys.argv[3]
    env = sys.argv[4]
    if len(sys.argv) > 5:
        user_name = sys.argv[5]
    else:
        user_name = ''

    server = ThreadedTCPServer((HOST, PORT), Handler)

    react_to_trace_thread = Thread(target=react_to_trace, args=(trace_file, running_machines_list, user_name, env))
    react_to_trace_thread.daemon = True
    react_to_trace_thread.start()

    with server:
        server.serve_forever()
