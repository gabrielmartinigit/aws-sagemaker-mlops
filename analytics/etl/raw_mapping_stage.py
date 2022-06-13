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

# Script generated for node Amazon S3
AmazonS3_node1653349812885 = glueContext.create_dynamic_frame.from_catalog(
    database="cgu-poc-raw",
    table_name="raw_passagem",
    transformation_ctx="AmazonS3_node1653349812885",
)

# Script generated for node AWS Glue Data Catalog
AWSGlueDataCatalog_node1654520716725 = glueContext.create_dynamic_frame.from_catalog(
    database="cgu-poc-raw",
    table_name="raw_feedbacks",
    transformation_ctx="AWSGlueDataCatalog_node1654520716725",
)

# Script generated for node AWS Glue Data Catalog
AWSGlueDataCatalog_node1653349246368 = glueContext.create_dynamic_frame.from_catalog(
    database="cgu-poc-raw",
    table_name="raw_pagamento",
    transformation_ctx="AWSGlueDataCatalog_node1653349246368",
)

# Script generated for node Apply Mapping
ApplyMapping_node1653349827156 = ApplyMapping.apply(
    frame=AmazonS3_node1653349812885,
    mappings=[
        (
            "identificador do processo de viagem",
            "long",
            "identificador_passagem",
            "string",
        ),
        ("`número da proposta (pcdp)`", "string", "numero_proposta", "string"),
        ("meio de transporte", "string", "meio_transporte", "string"),
        ("país - origem ida", "string", "pais_origem_ida", "string"),
        ("uf - origem ida", "string", "uf_origem", "string"),
        ("cidade - origem ida", "string", "cidade_origem_ida", "string"),
        ("país - destino ida", "string", "pais_destino_ida", "string"),
        ("uf - destino ida", "string", "uf_destino_ida", "string"),
        ("cidade - destino ida", "string", "cidade_destino_ida", "string"),
        ("país - origem volta", "string", "pais_origem_volta", "string"),
        ("uf - origem volta", "string", "uf_origem_volta", "string"),
        ("cidade - origem volta", "string", "cidade_origem_volta", "string"),
        ("pais - destino volta", "string", "pais_destino_volta", "string"),
        ("uf - destino volta", "string", "uf_destino_volta", "string"),
        ("cidade - destino volta", "string", "cidade_destino_volta", "string"),
        ("valor da passagem", "string", "valor_passagem", "string"),
        ("taxa de serviço", "string", "taxa_servico", "string"),
        ("data da emissão/compra", "string", "data_emissao", "string"),
        ("hora da emissão/compra", "string", "hora_emissao", "string"),
        ("year", "string", "year", "string"),
    ],
    transformation_ctx="ApplyMapping_node1653349827156",
)

# Script generated for node Apply Mapping
ApplyMapping_node1654520726141 = ApplyMapping.apply(
    frame=AWSGlueDataCatalog_node1654520716725,
    mappings=[
        ("col0", "string", "label", "string"),
        ("col1", "string", "comment", "string"),
        ("year", "string", "year", "string"),
    ],
    transformation_ctx="ApplyMapping_node1654520726141",
)

# Script generated for node Apply Mapping
ApplyMapping_node1653349253704 = ApplyMapping.apply(
    frame=AWSGlueDataCatalog_node1653349246368,
    mappings=[
        (
            "identificador do processo de viagem",
            "long",
            "identificador_passagem",
            "string",
        ),
        ("`número da proposta (pcdp)`", "string", "numero_proposta", "string"),
        ("código do órgão superior", "long", "codigo_orgao_superior", "string"),
        ("nome do órgão superior", "string", "nome_orgao_superior", "string"),
        ("codigo do órgão pagador", "long", "codigo_orgao_pagador", "string"),
        ("nome do órgao pagador", "string", "nome_orgao_pagador", "string"),
        (
            "código da unidade gestora pagadora",
            "long",
            "codigo_unidade_pagadora",
            "string",
        ),
        (
            "nome da unidade gestora pagadora",
            "string",
            "nome_unidade_pagadora",
            "string",
        ),
        ("tipo de pagamento", "string", "tipo_pagamento", "string"),
        ("valor", "string", "valor", "string"),
        ("year", "string", "year", "string"),
    ],
    transformation_ctx="ApplyMapping_node1653349253704",
)

# Script generated for node Amazon S3
AmazonS3_node1653349831088 = glueContext.getSink(
    path="s3://cgu-poc-stage/passagem/",
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=["year"],
    compression="snappy",
    enableUpdateCatalog=True,
    transformation_ctx="AmazonS3_node1653349831088",
)
AmazonS3_node1653349831088.setCatalogInfo(
    catalogDatabase="cgu-poc-stage", catalogTableName="stage_passagem"
)
AmazonS3_node1653349831088.setFormat("glueparquet")
AmazonS3_node1653349831088.writeFrame(ApplyMapping_node1653349827156)
# Script generated for node Amazon S3
AmazonS3_node1654520745603 = glueContext.getSink(
    path="s3://cgu-poc-stage/feedbacks/",
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=["year"],
    enableUpdateCatalog=True,
    transformation_ctx="AmazonS3_node1654520745603",
)
AmazonS3_node1654520745603.setCatalogInfo(
    catalogDatabase="cgu-poc-stage", catalogTableName="stage_feedbacks"
)
AmazonS3_node1654520745603.setFormat("csv")
AmazonS3_node1654520745603.writeFrame(ApplyMapping_node1654520726141)
# Script generated for node Amazon S3
AmazonS3_node1653349515336 = glueContext.getSink(
    path="s3://cgu-poc-stage/pagamento/",
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=["year"],
    compression="snappy",
    enableUpdateCatalog=True,
    transformation_ctx="AmazonS3_node1653349515336",
)
AmazonS3_node1653349515336.setCatalogInfo(
    catalogDatabase="cgu-poc-stage", catalogTableName="stage_pagamento"
)
AmazonS3_node1653349515336.setFormat("glueparquet")
AmazonS3_node1653349515336.writeFrame(ApplyMapping_node1653349253704)
job.commit()
