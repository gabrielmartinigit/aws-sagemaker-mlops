# SageMaker MLOps example
## Architecture

![Architecture](./images/mlops.jpg)

## Pre-reqs

- Python 3.8+
- AWS CLI
- CDK

## Getting Started

### Install Dependencies

```
git clone https://github.com/gabrielmartinigit/aws-sagemaker-mlops.git
cd aws-sagemaker-mlops/
pip3 install virtualenv
virtualenv .venv
source .venv/bin/activate
cd iac/
pip install -r requirements.txt
```

### Deploy Components

```
cdk ls
cdk synth
cdk bootstrap
cdk deploy --all --require-approval never
```

## Links & References

- AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
- CDK: https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html
- Amazon SageMaker workshop: https://catalog.us-east-1.prod.workshops.aws/workshops/63069e26-921c-4ce1-9cc7-dd882ff62575/
