#!/usr/bin/env python3
import os
import aws_cdk as cdk
from data_components.data_stack import DataStack
from mlops_components.mlops_stack import MLOpsStack

app = cdk.App()
DataStack(app, "DataStack")
MLOpsStack(app, "MLOpsStack")

app.synth()
