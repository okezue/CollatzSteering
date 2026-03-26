#!/bin/bash
set -e

KEY="okezue.pem"
REGION="us-east-1"

usage() {
    echo "Usage:"
    echo "  $0 launch [g5.xlarge|p3.2xlarge|...]"
    echo "  $0 setup <host>"
    echo "  $0 run <host> <base> [extra args]"
    echo "  $0 sync <host> [up|down]"
    exit 1
}

launch_instance() {
    ITYPE=${1:-g5.xlarge}
    echo "Launching $ITYPE..."
    IID=$(aws ec2 run-instances \
        --region $REGION \
        --image-id ami-0c7217cdde317cfec \
        --instance-type $ITYPE \
        --key-name okezue \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=collatz-steering}]' \
        --query 'Instances[0].InstanceId' --output text)
    echo "Instance: $IID"
    echo "Waiting for running state..."
    aws ec2 wait instance-running --instance-ids $IID --region $REGION
    IP=$(aws ec2 describe-instances --instance-ids $IID --region $REGION \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    echo "IP: $IP"
    echo "SSH: ssh -i $KEY ubuntu@$IP"
}

setup_instance() {
    HOST=$1
    echo "Setting up $HOST..."
    ssh -i $KEY ubuntu@$HOST << 'SETUP'
sudo apt-get update -qq
pip install --upgrade pip
pip install torch numpy matplotlib tqdm wandb
mkdir -p ~/CollatzSteering
SETUP
    echo "Uploading code..."
    scp -i $KEY -r ./*.py ./requirements.txt ubuntu@$HOST:~/CollatzSteering/
    echo "Setup complete."
}

run_training() {
    HOST=$1;BASE=$2;shift 2
    echo "Running base=$BASE on $HOST..."
    ssh -i $KEY ubuntu@$HOST "cd ~/CollatzSteering && nohup python run.py train --base $BASE $@ > train_b${BASE}.log 2>&1 &"
    echo "Training started. Monitor with:"
    echo "  ssh -i $KEY ubuntu@$HOST 'tail -f ~/CollatzSteering/train_b${BASE}.log'"
}

sync_results() {
    HOST=$1;DIR=${2:-down}
    if [ "$DIR" = "down" ]; then
        echo "Downloading results..."
        scp -i $KEY -r ubuntu@$HOST:~/CollatzSteering/output/ ./output/
    else
        echo "Uploading code..."
        scp -i $KEY -r ./*.py ./requirements.txt ubuntu@$HOST:~/CollatzSteering/
    fi
}

case ${1:-} in
    launch) launch_instance $2;;
    setup) setup_instance $2;;
    run) run_training $2 $3 ${@:4};;
    sync) sync_results $2 $3;;
    *) usage;;
esac
