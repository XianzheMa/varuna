import sys

from kubernetes import client, config
import os
import time
import argparse
# it should be executed from the parent directory of "run"
if __name__ == '__main__':
    config.load_config()
    v1 = client.CoreV1Api()

    parser = argparse.ArgumentParser(description='Trace controller')

    parser.add_argument('--no_training', action='store_true', help="whether to train")
    parser.add_argument('--no_profiling', action='store_true', help="whether to profile")
    parser.add_argument('--no_machine_list_update', action='store_true', help="whether to update machine list")
    parser.add_argument('--num_nodes', type=int, help="maximal number of nodes in the cluster")
    args = parser.parse_args()

    if not args.no_machine_list_update:
        assert args.num_nodes is not None, "please specify the number of nodes"
        while True:
            pods = v1.list_pod_for_all_namespaces(
                    label_selector="app=elastic-ml-worker",
                    field_selector="status.phase=Running",
                    watch=False).items
            if len(pods) != args.num_nodes:
                print(f"Waiting for {args.num_nodes} pods to be ready")
                time.sleep(3)
                continue
            else:
                break

        # to run inside a cluster, the ip address is replaced by the pod name
        # as kubectl exec works with pod name
        pod_names = [p.metadata.name for p in pods]
        with open('machine_list.txt', 'w') as f:
            f.write('\n'.join(pod_names))


    if not args.no_profiling:
        # profile the model
        start = time.time()
        os.system("bash profile_gpt.sh")

        end = time.time()
        print(f'it takes {end - start} secs to profile the model')

    if not args.no_training:
        # launch varuna
        os.system("bash train_gpt.sh")
