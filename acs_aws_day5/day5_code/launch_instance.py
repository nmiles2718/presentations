#!/usr/bin/env python
import argparse
import boto3


parser = argparse.ArgumentParser()
parser.add_argument(
    '-template_id',
    help='Launch Template ID'
)
parser.add_argument(
    '-keyname',
    help='SSH key name (exclude .pem)'
)

def launch_instance(template_id=None, keyname=None):
    template_id=template_id
    s = boto3.Session(region_name='us-east-1', profile_name=None)
    ec2 = s.resource('ec2')
    LaunchTemplate = {'LaunchTemplateId': template_id}
    instances = ec2.create_instances(
        LaunchTemplate=LaunchTemplate,
        MaxCount=1,
        MinCount=1,
        KeyName=keyname
    )

    instance = instances[0]

    print('Launched EC2 instance {}'.format(instance.id))
    instance.wait_until_running()
    instance.load()
    print(f"EC2 DNS: {instance.public_dns_name}")


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    launch_instance(**args)