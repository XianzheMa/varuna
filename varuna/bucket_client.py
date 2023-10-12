from google.cloud import storage
import os
class BucketClient:

    def __init__(self, bucket_name):
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_name)

    def upload_profiles(self, profile_dir, remote_profile_dir):
        for file in os.listdir(profile_dir):
            blob = self.bucket.blob(os.path.join(remote_profile_dir, file))
            blob.upload_from_filename(os.path.join(profile_dir, file), timeout=None)

    def download_profiles(self, profile_dir, remote_profile_dir):
        os.makedirs(profile_dir, exist_ok=True)
        blobs = self.bucket.list_blobs(prefix=remote_profile_dir)
        for blob in blobs:
            simple_name = blob.name.split('/')[-1]
            local_file = os.path.join(profile_dir, simple_name)
            print(f'download {blob.name} to {local_file}')
            blob.download_to_filename(local_file, timeout=None)

    def upload_file(self, local_path, remote_path):
        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(local_path, timeout=None)

    def download_file(self, local_path, remote_path):
        blob = self.bucket.blob(remote_path)
        blob.download_to_filename(local_path, timeout=None)

    def write_to_remote(self, content, remote_path):
        blob = self.bucket.blob(remote_path)
        blob.upload_from_string(content, timeout=None)

    def read_from_remote(self, remote_path):
        blob = self.bucket.blob(remote_path)
        return blob.download_as_string(timeout=None)
