import os
import requests
import tarfile
import argparse
from tqdm import tqdm

def download_and_process_tar_files(start_num=155, end_num=2480, max_images=None, base_url="https://huggingface.co/datasets/Spawning/pd12m-full/resolve/main/", data_dir="data"):
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    total_images = 0
    current_num = start_num
    
    # Setup progress bar for the range of files
    pbar = tqdm(total=(end_num - start_num + 1), desc="Processing tar files")
    
    while current_num <= end_num:
        # Format the file number with leading zeros (5 digits)
        file_num = f"{current_num:05d}"
        
        # Construct the full URL
        url = f"{base_url}{file_num}.tar"
        
        # Construct the local file path
        local_file = os.path.join(data_dir, f"{file_num}.tar")
        
        try:
            # Download the file if it doesn't exist
            if not os.path.exists(local_file):
                print(f"Downloading {url} to {local_file}")
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                with open(local_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Process the tar file to count files
            with tarfile.open(local_file) as tar:
                file_count = len(tar.getmembers())
            
            # Calculate number of images (files divided by 3)
            images_in_tar = file_count // 3
            total_images += images_in_tar
            
            print(f"File {file_num}.tar contains {file_count} files ({images_in_tar} images)")
            print(f"Total images so far: {total_images}")
            
            # If we've reached or exceeded max_images, break
            if max_images is not None and total_images >= max_images:
                print(f"Reached maximum number of images ({max_images})")
                break
                
        except Exception as e:
            print(f"Error processing {url}: {e}")
        
        # Move to the next file
        current_num += 1
        pbar.update(1)
    
    pbar.close()
    print(f"Downloaded files from {start_num:05d}.tar to {current_num-1:05d}.tar")
    print(f"Total images: {total_images}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and process tar files from Hugging Face dataset.')
    parser.add_argument('--start', type=int, default=155, help='Starting file number')
    parser.add_argument('--end', type=int, default=2480, help='Ending file number')
    parser.add_argument('--max-images', type=int, default=None, help='Maximum number of images to download')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to save the tar files')
    
    args = parser.parse_args()
    
    download_and_process_tar_files(
        start_num=args.start,
        end_num=args.end,
        max_images=args.max_images,
        data_dir=args.data_dir
    )