import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Script generated for node AWS Glue Data Catalog
AWSGlueDataCatalog_node1654521951654 = glueContext.create_dynamic_frame.from_catalog(
    database="cgu-poc-stage",
    table_name="stage_feedbacks",
    transformation_ctx="AWSGlueDataCatalog_node1654521951654",
)

AWSGlueDataCatalog_node1654521951654=AWSGlueDataCatalog_node1654521951654.coalesce(1)

# Script generated for node AWS Glue Data Catalog
AWSGlueDataCatalog_node1654570272579 = glueContext.create_dynamic_frame.from_catalog(
    database="cgu-poc-stage",
    table_name="stage_passagem",
    transformation_ctx="AWSGlueDataCatalog_node1654570272579",
)

# Script generated for node Apply Mapping
ApplyMapping_node1654521962353 = ApplyMapping.apply(
    frame=AWSGlueDataCatalog_node1654521951654,
    mappings=[
        ("col0", "string", "label", "string"),
        ("col1", "string", "comment", "string"),
        ("year", "string", "year", "string"),
    ],
    transformation_ctx="ApplyMapping_node1654521962353",
)

# Script generated for node Apply Mapping
ApplyMapping_node1654570285754 = ApplyMapping.apply(
    frame=AWSGlueDataCatalog_node1654570272579,
    mappings=[
        ("identificador_passagem", "string", "identificador_passagem", "string"),
        ("numero_proposta", "string", "numero_proposta", "string"),
        ("meio_transporte", "string", "meio_transporte", "string"),
        ("pais_origem_ida", "string", "pais_origem_ida", "string"),
        ("uf_origem", "string", "uf_origem", "string"),
        ("cidade_origem_ida", "string", "cidade_origem_ida", "string"),
        ("pais_destino_ida", "string", "pais_destino_ida", "string"),
        ("uf_destino_ida", "string", "uf_destino_ida", "string"),
        ("cidade_destino_ida", "string", "cidade_destino_ida", "string"),
        ("pais_origem_volta", "string", "pais_origem_volta", "string"),
        ("uf_origem_volta", "string", "uf_origem_volta", "string"),
        ("cidade_origem_volta", "string", "cidade_origem_volta", "string"),
        ("pais_destino_volta", "string", "pais_destino_volta", "string"),
        ("uf_destino_volta", "string", "uf_destino_volta", "string"),
        ("cidade_destino_volta", "string", "cidade_destino_volta", "string"),
        ("valor_passagem", "string", "valor_passagem", "string"),
        ("taxa_servico", "string", "taxa_servico", "string"),
        ("data_emissao", "string", "data_emissao", "string"),
        ("hora_emissao", "string", "hora_emissao", "string"),
        ("year", "string", "year", "string"),
    ],
    transformation_ctx="ApplyMapping_node1654570285754",
)

# Script generated for node Amazon S3
AmazonS3_node1654521968116 = glueContext.getSink(
    path="s3://cgu-poc-analytics/feedbacks/",
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=["year"],
    enableUpdateCatalog=True,
    transformation_ctx="AmazonS3_node1654521968116",
)
AmazonS3_node1654521968116.setCatalogInfo(
    catalogDatabase="cgu-poc-analytics", catalogTableName="analytics_feedbacks"
)
AmazonS3_node1654521968116.setFormat("csv")
AmazonS3_node1654521968116.writeFrame(ApplyMapping_node1654521962353)
# Script generated for node Amazon S3
AmazonS3_node1654570291458 = glueContext.getSink(
    path="s3://cgu-poc-analytics/passagem/",
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=["year"],
    compression="snappy",
    enableUpdateCatalog=True,
    transformation_ctx="AmazonS3_node1654570291458",
)
AmazonS3_node1654570291458.setCatalogInfo(
    catalogDatabase="cgu-poc-analytics", catalogTableName="analytics_passagem"
)
AmazonS3_node1654570291458.setFormat("glueparquet")
AmazonS3_node1654570291458.writeFrame(ApplyMapping_node1654570285754)
job.commit()
