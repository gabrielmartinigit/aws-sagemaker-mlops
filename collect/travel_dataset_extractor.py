import sys
import time
# from awsglue.utils import getResolvedOptions
import requests
import zipfile
import io
import boto3  # AWS SDK for S3 interactions

PORTAL_URL = "https://www.portaltransparencia.gov.br/download-de-dados/viagens"
MAX_RETRIES = 2
SEC_RATE_LIMIT = 10
BUCKET = "cgu-poc-raw"
TABLES = ["pagamento", "passagem", "trecho", "viagem"]

client_s3 = boto3.client("s3")


def extractor(years=["2019", "2021", "2022"]):
    for year in years:
        print(f"Extracting: {year}")
        year_response = requests.get(f"{PORTAL_URL}/{year}")
        print(f"Status: {year_response.status_code}")

        # Retry loop if portal returns 50X or 40X
        retries = 1
        while(year_response.status_code and
              year_response.status_code != 200 and
              retries <= MAX_RETRIES):
            time.sleep(SEC_RATE_LIMIT)  # wait X senconds to bypass rate limit
            print(f"Tentative: {retries} of {MAX_RETRIES}")
            print(f"Status: {year_response.status_code}")
            retries += 1

        if year_response.status_code == 200:
            year_zip_file = zipfile.ZipFile(io.BytesIO(year_response.content))
            # Unzip and store CSV
            with year_zip_file as zip_object:
                list_of_files = zip_object.namelist()
                for file_name in list_of_files:
                    if file_name.endswith('.csv'):
                        # Check tables based in file name
                        table = [table for table in TABLES
                                 if(table in file_name.lower())][0]
                        # Extract a single file from zip and upload to s3
                        client_s3.put_object(
                            Body=zip_object.open(file_name).read(),
                            Bucket=BUCKET,
                            Key=f'{table}/year={year}/{file_name}')
                        print(f"Uploaded: {file_name}")

if __name__ == "__main__":
    ''' args = getResolvedOptions(sys.argv,['years'])
    # Verify years parameter from Glue job
    extractor(args['years']) if args['years'] else extractor()'''
    extractor()
