#!/bin/bash
# the script to kill the varuna processes on the all machines in the machine list

ip_file=$1
env=$2
# if the number of arguments is 3, then the third argument is the username
if [ $# -eq 3 ]; then
	username=$3
else
	username=""
fi
machines=($(cat $ip_file))
nservers=${#machines[@]}

i=0
while [ $i -lt $nservers ]
do
    echo $i ${machines[i]}
    server=${machines[i]}

	# if env is k8s
	if [ $env = "k8s" ]; then
		kubectl exec $server -- /bin/bash -c 'pkill -f gpt_script; kill $(lsof -t -i:29500)'
	else
		if [ $server = "127.0.0.1" ]; then
			pkill -f gpt_script
			kill $(lsof -t -i:29500)
		else
			# if username is not an empty string
			if [ -z "$username" ]; then
				full_server=$server
			else
				full_server=$username@$server
			fi
			echo "full server name is $full_server"
			ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 $full_server 'pkill -f gpt_script; kill $(lsof -t -i:29500)'
		fi
	fi
    i=$(($i+1))
done
