#!/usr/bin/env python3
import os
import aws_cdk as cdk
from science_components.science_stack import ScienceStack

app = cdk.App()
ScienceStack(app, "ScienceStack")

app.synth()
