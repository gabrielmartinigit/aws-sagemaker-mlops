#!/usr/bin/env python3
import os
import aws_cdk as cdk
from data_components.data_stack import DataStack
from mlops_components.mlops_stack import MLOpsStack
from science_components.science_stack import ScienceStack

app = cdk.App()
DataStack(app, "DataStack")
MLOpsStack(app, "MLOpsStack")
ScienceStack(app, "ScienceStack")

app.synth()
