"""Set MinIO bucket policy to allow public read access"""
import os
import json
from dotenv import load_dotenv
load_dotenv("../../.env.dev")

from minio import Minio

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = "animal-images"

def set_bucket_public_policy():
    """Set bucket policy to allow public read access"""
    
    # Initialize MinIO client
    client = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    
    # Define policy - allow public read access to all objects in bucket
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{BUCKET_NAME}/*"]
            }
        ]
    }
    
    try:
        # Set the bucket policy
        client.set_bucket_policy(BUCKET_NAME, json.dumps(policy))
        print(f"✅ Successfully set public read policy for bucket: {BUCKET_NAME}")
        print(f"   Images are now publicly accessible")
        
        # Verify policy
        current_policy = client.get_bucket_policy(BUCKET_NAME)
        print(f"\n📋 Current policy:")
        print(json.dumps(json.loads(current_policy), indent=2))
        
    except Exception as e:
        print(f"❌ Error setting bucket policy: {str(e)}")

if __name__ == "__main__":
    set_bucket_public_policy()
