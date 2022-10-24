from aws_cdk import (
    Stack,
    aws_sagemaker as sagemaker
)
from constructs import Construct


class ScienceStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # SageMaker studio, permissions