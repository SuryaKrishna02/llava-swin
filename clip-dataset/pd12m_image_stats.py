import os
import tarfile
import argparse
import mimetypes

def count_images_in_folder(folder_path, exclude_files=None):
    """
    Count the number of images in .tar files within the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing .tar files
        exclude_files (list): List of filenames to exclude from the final count
        
    Returns:
        tuple: (Total image count, Dictionary of image counts per tar file, Excluded image count)
    """
    total_images = 0
    excluded_images = 0
    image_counts_per_file = {}
    
    # Default to empty list if None
    if exclude_files is None:
        exclude_files = []
    
    # Ensure mimetypes are initialized
    mimetypes.init()
    
    # Get all .tar files in the folder
    tar_files = [f for f in os.listdir(folder_path) if f.endswith('.tar')]
    
    print(f"Found {len(tar_files)} .tar files in the folder")
    
    # Process each .tar file
    for tar_file in tar_files:
        tar_path = os.path.join(folder_path, tar_file)
        image_count = 0
        
        try:
            # Open the tar file
            with tarfile.open(tar_path) as tar:
                # Count files in the tar
                total_files = len(tar.getmembers())
                
                # Extract and check each file
                for member in tar.getmembers():
                    # Skip directories
                    if member.isdir():
                        continue
                    
                    # Get file extension
                    file_ext = os.path.splitext(member.name)[1].lower()
                    
                    # Check if it's an image based on extension or mimetype
                    is_image = file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
                    
                    if not is_image and file_ext:
                        # Try to determine mimetype
                        mime_type = mimetypes.guess_type(member.name)[0]
                        is_image = mime_type and mime_type.startswith('image/')
                    
                    if is_image:
                        image_count += 1
            
            print(f"Tar file '{tar_file}' contains {image_count} images out of {total_files} files")
            
            # Store the count for this tar file
            image_counts_per_file[tar_file] = image_count
            
            # Add to total or excluded counts based on filename
            if tar_file in exclude_files:
                excluded_images += image_count
                print(f"  -> Excluding {image_count} images from '{tar_file}'")
            else:
                total_images += image_count
            
        except Exception as e:
            print(f"Error processing '{tar_file}': {str(e)}")
    
    return total_images, image_counts_per_file, excluded_images

def main():
    parser = argparse.ArgumentParser(description='Count images in .tar files')
    parser.add_argument('folder_path', help='Path to the folder containing .tar files')
    parser.add_argument('--exclude', '-e', nargs='+', help='List of .tar files to exclude from the count')
    
    args = parser.parse_args()
    folder_path = args.folder_path
    exclude_files = args.exclude or []
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory")
        return
    
    total, per_file, excluded = count_images_in_folder(folder_path, exclude_files)
    
    # Print detailed results
    print("\n----- Image Count Summary -----")
    print("Individual file counts:")
    for tar_file, count in per_file.items():
        status = "(Excluded)" if tar_file in exclude_files else ""
        print(f"  - {tar_file}: {count} images {status}")
    
    print(f"\nTotal images (including excluded files): {total + excluded}")
    if excluded > 0:
        print(f"Total excluded images: {excluded}")
        print(f"Final image count (after exclusions): {total}")
    else:
        print(f"No files were excluded.")

# Execute the main function if script is run directly
if __name__ == "__main__":
    main()