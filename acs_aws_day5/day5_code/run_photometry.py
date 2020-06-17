# Import astroquery.mast (http://astroquery.readthedocs.io/en/latest/mast/mast.html)
# Note, you may need to build from source to access the HST data on AWS.
import argparse
import json
import time

import boto3
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-catalog',
                    default=None,
                    help='catalog of files to process')
parser.add_argument('-lambda_name',
                    default=None,
                    help='name of the lambda function')
parser.add_argument('-output_bucket',
                    default=None,
                    help='name of output bucket')


def process_catalog(catalog=None, lambda_name=None, output_bucket=None):
    path_urls = [val.strip('\n') for val in open(catalog).readlines()]
    # Auth to create a Lambda function
    session = boto3.Session(profile_name=None)
    client = session.client('lambda', region_name='us-east-1')
    delayed_objs = []
    st = time.time()
    for url in tqdm(path_urls[:10]):
        event = {
            'fits_s3_key': url,
            'fits_s3_bucket': 'stpubdata',
            'radius':3,
            'r_inner': 14,
            'r_outer': 16,
            's3_output_bucket': output_bucket
        }

        Payload = json.dumps(event)
        lambda_inputs = {
            'FunctionName': lambda_name,
            'InvocationType': 'Event',
            'LogType': 'Tail',
            'Payload': Payload
        }
        response = client.invoke(**lambda_inputs)
    et = time.time()
    print(f"Duration: {et - st:0.2f}")

if __name__ == "__main__":
    args = parser.parse_args()
    args = vars(args)
    if args['catalog'] is not None and \
            args['lambda_name'] is not None and \
            args['output_bucket']:
        process_catalog(**args)
