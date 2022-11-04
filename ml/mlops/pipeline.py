import sagemaker
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString
)
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline

# Pipeline parameters
training_instance_type = ParameterString(
    name="TrainingInstanceType",
    default_value="ml.c5.xlarge"
)

model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval"
)

model_package_group_name = "TitanicModelPackageGroupName"

# Train step
FRAMEWORK_VERSION = "0.23-1"

sklearn_estimator = SKLearn(
    entry_point="train_inference.py",
    source_dir="../train_inference",
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type=training_instance_type,
    framework_version=FRAMEWORK_VERSION,
    base_job_name="titanic-scikit",
    metric_definitions=[{"Name": "Accuracy",
                         "Regex": "Accuracy: ([0-9.]+).*$"}],
    hyperparameters={
        "n-estimators": 100,
        "features": "sex age",
        "target": "survived",
    },
)

step_train = TrainingStep(
    name="TitanicTrain",
    estimator=sklearn_estimator,
    inputs={
        "train": TrainingInput(
            s3_data="s3://mlopsstack-mlbucket12760f44-590xj4q47o7h/datasets/titanic/train/train.csv",
            content_type="text/csv"
        ),
        "test": TrainingInput(
            s3_data="s3://mlopsstack-mlbucket12760f44-590xj4q47o7h/datasets/titanic/test/test.csv",
            content_type="text/csv"
        )
    },
)

# Register step
step_register = RegisterModel(
    name="AbaloneRegisterModel",
    estimator=sklearn_estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    model_package_group_name=model_package_group_name,
    approval_status=model_approval_status
)

# Create pipeline
pipeline_name = "titanic-pipeline"

# Combine pipeline steps and create pipeline
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        training_instance_type,
        model_approval_status
    ],
    steps=[
        step_train,
        step_register
    ],
)

pipeline.upsert(role_arn=sagemaker.get_execution_role())

# https://docs.aws.amazon.com/sagemaker/latest/dg/define-pipeline.html
