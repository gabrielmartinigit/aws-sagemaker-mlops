import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import (ParameterString)
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline

# Pipeline parameters
pipeline_session = PipelineSession()

input_data = ParameterString(
    name="InputData"
)

training_instance_type = ParameterString(
    name="TrainingInstanceType",
    default_value="ml.c5.xlarge"
)

model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval"
)

model_package_group_name = "TitanicModelPackageGroupName"

FRAMEWORK_VERSION = "0.23-1"

# Processing step
sklearn_processor = SKLearnProcessor(
    framework_version=FRAMEWORK_VERSION,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="sklearn-titanic-process",
    role=sagemaker.get_execution_role(),
)

step_process = ProcessingStep(
    name="TitanicProcess",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(source=input_data,
                        destination="/opt/ml/processing/input"),
    ],
    outputs=[
        ProcessingOutput(output_name="train",
                         source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="validation",
                         source="/opt/ml/processing/validation"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
    ],
    code="../prepare/prepare.py",
)

# Train step
sklearn_estimator = SKLearn(
    entry_point="train_inference.py",
    source_dir="../train_inference",
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type=training_instance_type,
    framework_version=FRAMEWORK_VERSION,
    py_version='py3',
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
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "validation"
            ].S3Output.S3Uri,
            content_type="text/csv"
        )
    },
)

# Create model
sklearn_model = SKLearnModel(
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    entry_point='train_inference.py',
    source_dir='../train_inference',
    framework_version=FRAMEWORK_VERSION,
    role=sagemaker.get_execution_role(),
    sagemaker_session=pipeline_session,
    py_version='py3'
)

# Register step
register_model_step_args = sklearn_model.register(
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    model_package_group_name=model_package_group_name,
    approval_status=model_approval_status
)

step_register = ModelStep(
    name="TitanicRegisterModel",
    step_args=register_model_step_args,
)

# Create pipeline
pipeline_name = "titanic-pipeline"

# Combine pipeline steps and create pipeline
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        input_data,
        training_instance_type,
        model_approval_status
    ],
    steps=[
        step_process,
        step_train,
        step_register
    ],
)

pipeline.upsert(role_arn=sagemaker.get_execution_role())

# https://docs.aws.amazon.com/sagemaker/latest/dg/define-pipeline.html
