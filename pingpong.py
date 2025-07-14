import requests
from pathlib import Path
import boto3

# AWS S3 bucket and client
bucket_name = "dataset"
s3 = boto3.client("s3")

# Base URL for downloads
base_url = "https://someurl.com/"

# List of files to download and upload
urls = [
    # Games
    base_url + "game_1.mp4",
    base_url + "game_1.zip",
    base_url + "test_1.mp4",
    base_url + "test_1.zip"
]


def determine_s3_key(filename: str) -> str:
    """Return the correct S3 key based on filename and type."""
    if filename.startswith("game_"):
        base_dir = "trainingdata"
    elif filename.startswith("test_"):
        base_dir = "testdata"
    else:
        raise ValueError(f"Unexpected filename pattern: {filename}")

    if filename.endswith(".mp4"):
        sub_dir = "video"
    elif filename.endswith(".zip"):
        sub_dir = "markup"
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    return f"{base_dir}/{sub_dir}/{filename}"


def download_and_upload_to_s3(url):
    filename = url.split("/")[-1]
    local_path = Path("/tmp") / filename

    # Download file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"✅ Downloaded: {filename}")

    # Determine destination key and upload
    s3_key = determine_s3_key(filename)
    s3.upload_file(str(local_path), bucket_name, s3_key)
    print(f"🚀 Uploaded to s3://{bucket_name}/{s3_key}")
