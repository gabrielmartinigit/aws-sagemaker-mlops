CREATE external schema spectrum_schema FROM data catalog 
database 'analytics-datalake' 
iam_role 'arn:aws:iam::ACCOUNT:role/RedshiftRoleSpectrum';

SELECT * FROM "dev"."spectrum_schema"."analytics_passagem";

CREATE MATERIALIZED VIEW analytics_view
AS SELECT * FROM "dev"."spectrum_schema"."analytics_passagem";

SELECT * FROM analytics_view;

