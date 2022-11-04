#!/usr/bin/env python3
import os
import boto3
import aws_cdk as cdk
from data_components.data_stack import DataStack
from mlops_components.mlops_stack import MLOpsStack
from science_components.science_stack import ScienceStack

sts_client = boto3.client("sts")
account_id = os.environ.get(
    'CDK_DEFAULT_ACCOUNT',
    sts_client.get_caller_identity()["Account"]
)
region = os.environ.get('CDK_DEFAULT_REGION', 'us-east-1')

app = cdk.App()
DataStack(app, "DataStack")
MLOpsStack(app, "MLOpsStack")
ScienceStack(app,
             "ScienceStack",
             env={
                 "account": account_id,
                 'region': region
             })

app.synth()
