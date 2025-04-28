import os
import shutil
import tarfile
import requests
import argparse
from PIL import Image
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def resize_image(img_path):
    """Resize a single image to 192x192."""
    try:
        with Image.open(img_path) as img:
            resized_img = img.resize((192, 192), Image.LANCZOS)
            resized_img.save(img_path)
        return True
    except Exception as e:
        print(f"Error resizing {img_path}: {e}")
        return False

def download_and_process_tar_files(start_num=155, end_num=2480, max_images=None, base_url="https://huggingface.co/datasets/Spawning/pd12m-full/resolve/main/", data_dir="data"):
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    total_images = 0
    current_num = start_num
    
    # Setup progress bar for the range of files
    pbar = tqdm(total=(end_num - start_num + 1), desc="Processing tar files")
    
    # Get number of CPU cores for multiprocessing
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for image resizing")
    
    while current_num <= end_num:
        # Format the file number with leading zeros (5 digits)
        file_num = f"{current_num:05d}"
        
        # Construct the full URL
        url = f"{base_url}{file_num}.tar"
        
        # Construct the local file path
        local_file = os.path.join(data_dir, f"{file_num}.tar")
        
        # Create a temporary extraction directory
        extract_dir = os.path.join(data_dir, f"temp_{file_num}")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            # Download the file if it doesn't exist
            if not os.path.exists(local_file):
                print(f"Downloading {url} to {local_file}")
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                with open(local_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Extract the tar file and track all files
            all_files = []
            image_files = []
            
            print(f"Extracting {local_file} to {extract_dir}")
            with tarfile.open(local_file) as tar:
                members = tar.getmembers()
                for member in members:
                    tar.extract(member, path=extract_dir)
                    all_files.append(member.name)
                    # Check if file is an image that needs resizing
                    if member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(os.path.join(extract_dir, member.name))
                file_count = len(members)
            
            # Delete the original tar file to save space
            os.remove(local_file)
            print(f"Deleted original tar file: {local_file}")
            
            # Calculate number of images
            images_in_tar = len(image_files)
            total_images += images_in_tar
            
            # Resize all images to 192x192 using multiprocessing
            print(f"Resizing {images_in_tar} images in {extract_dir} to 192x192 using multiprocessing")
            
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                list(tqdm(executor.map(resize_image, image_files), 
                          total=len(image_files), 
                          desc=f"Resizing images in {file_num}"))
            
            # Create a new tar file with all files (including resized images)
            new_tar_file = os.path.join(data_dir, f"{file_num}.tar")
            print(f"Creating new tar file: {new_tar_file}")
            with tarfile.open(new_tar_file, "w") as tar:
                # Change to the extract directory to avoid full paths in tar
                original_dir = os.getcwd()
                os.chdir(extract_dir)
                
                # Add all files to the tar
                for root, dirs, files in os.walk('.'):
                    for file in files:
                        tar.add(os.path.join(root, file))
                
                # Change back to the original directory
                os.chdir(original_dir)
            
            # Delete the temporary extraction directory
            shutil.rmtree(extract_dir)
            print(f"Deleted temporary directory: {extract_dir}")
            
            print(f"File {file_num}.tar processed: {file_count} files ({images_in_tar} images)")
            print(f"Total images processed so far: {total_images}")
            
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
    print(f"Downloaded and processed files from {start_num:05d}.tar to {current_num-1:05d}.tar")
    print(f"Total images: {total_images}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download, process, resize, and repack tar files from Hugging Face dataset.')
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