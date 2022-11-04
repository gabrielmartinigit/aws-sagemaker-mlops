from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_iam as iam,
    aws_glue as glue
)
from constructs import Construct


class DataStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        data_lake_bucket = s3.Bucket(
            self,
            "TrustedBucket"
        )

        athena_bucket = s3.Bucket(
            self,
            "AthenaBucket"
        )

        s3deploy.BucketDeployment(
            self,
            "DeployData",
            sources=[
                s3deploy.Source.asset("../dataset/")
            ],
            destination_bucket=data_lake_bucket,
            destination_key_prefix="titanic"
        )

        glue_role = iam.Role(
            self,
            "GlueRole",
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSGlueServiceRole")
            ]
        )

        data_lake_bucket.grant_read_write(glue_role)

        cfn_crawler = glue.CfnCrawler(
            self,
            "TitanicCrawler",
            role=glue_role.role_arn,
            targets=glue.CfnCrawler.TargetsProperty(
                s3_targets=[glue.CfnCrawler.S3TargetProperty(
                    path=f"s3://{data_lake_bucket.bucket_name}/titanic/"
                )]
            ),
            database_name="titanic_db"
        )
