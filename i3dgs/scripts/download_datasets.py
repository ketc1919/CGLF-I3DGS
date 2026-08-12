import os
import urllib.request
import zipfile
import argparse
from tqdm import tqdm

def download_and_extract(url, extract_to):
    """
    Download a zip file from a URL and extract its contents to a specified directory.

    Args:
        url (str): The URL of the zip file to download.
        extract_to (str): The directory where the contents will be extracted.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Define the local filename
    local_filename = os.path.join(extract_to, os.path.basename(url))

    # Download the file
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
              desc=f"Downloading {os.path.basename(url)}") as pbar:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pbar.total = total_size
                downloaded = min(downloaded, total_size)
            pbar.update(downloaded - pbar.n)
        urllib.request.urlretrieve(url, local_filename, reporthook)

    # Extract the zip file
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=f"Extracting {os.path.basename(local_filename)}"):
            zip_ref.extract(member, extract_to)
    # Remove the zip file after extraction
    os.remove(local_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract datasets.")
    parser.add_argument("--datasets", nargs='+', default=["TUM", "MipNeRF360", "MipNeRF360U", "StaticHikes", "tandt_db"],
                        help="List of datasets to download. Default is TUM, MipNeRF360, MipNeRF360U, StaticHikes, tandt_db.")
    parser.add_argument("--out_dir", type=str, default="data", 
                        help="Output base directory for the datasets.")
    args = parser.parse_args()
    
    # Define the base URL and dataset names
    base_url = "https://repo-sam.inria.fr/nerphys/on-the-fly-nvs/datasets"

    # Datasets hosted at a different location
    external_urls = {
        "tandt_db": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip",
    }

    # Download and extract each dataset
    for dataset in args.datasets:
        url = external_urls.get(dataset, f"{base_url}/{dataset}.zip")
        download_and_extract(url, args.out_dir)
    
    print("Done")