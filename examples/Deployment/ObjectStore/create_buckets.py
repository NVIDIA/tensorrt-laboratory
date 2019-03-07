import os
import boto3

s3 = boto3.client("s3", use_ssl=False, verify=False, 
                  endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))

response = s3.list_buckets()

buckets = [b["Name"] for b in response["Buckets"]]

if "images" not in buckets:
    s3.create_bucket(Bucket="images")

response = s3.list_buckets()
buckets = [b["Name"] for b in response["Buckets"]]

print(buckets)
