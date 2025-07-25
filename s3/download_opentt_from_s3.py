import boto3
import os
import zipfile
from pathlib import Path

# Config
bucket_name = "pingpongdataset"
prefixes = ["trainingdata/", "testdata/"]
local_base = Path("/opt/dlami/nvme/opentt_data")

# Initialize S3 client
s3 = boto3.client("s3")

def download_and_extract(key, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading s3://{bucket_name}/{key} → {local_path}")
    s3.download_file(bucket_name, key, str(local_path))

    # If it's a .zip, extract it
    if local_path.suffix == ".zip":
        print(f"📦 Extracting {local_path.name}")
        with zipfile.ZipFile(local_path, "r") as zip_ref:
            zip_ref.extractall(local_path.parent / local_path.stem)

for prefix in prefixes:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):  # Skip folder markers
                continue

            # Construct full local path
            relative_path = key  # this preserves trainingdata/markup/... structure
            local_path = local_base / relative_path
            download_and_extract(key, local_path)

print("✅ Done downloading and extracting.")