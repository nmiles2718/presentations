#!/usr/bin/env bash

if [[ $# -lt 2 ]] ; then
    echo "Not enough arguments supplied"
    echo "Required Positional Arguments:"
    echo "1. EC2 public DNS"
    echo "2. EC2 private ssh key"
    exit 1
fi

# Copy credentials to EC2 instanace
ec2_ip=$1  # first arg is IP address of instance
keyname=$2
aws_dir=~/.aws/
ssh_credentials=~/.ssh/
aws_ssh_key=$aws_dir$keyname

# Copy files credentials to AWS
scp -ri "$aws_ssh_key" $aws_dir ec2-user@$ec2_ip:~
scp -ri "$aws_ssh_key" $ssh_credentials ec2-user@$ec2_ip:~
scp -ri "$aws_ssh_key" ~/aws_setup/install_miniconda3_linux.sh ec2-user@$ec2_ip:~

# log in to the instance
ssh -i "$aws_ssh_key" ec2-user@$ec2_ip