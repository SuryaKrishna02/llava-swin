import io
import tarfile
from PIL import Image, ImageDraw

def create_dummy_image(text, size=(256, 256), bgcolor="white"):
    """Creates a dummy image with text."""
    img = Image.new("RGB", size, bgcolor)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill="black")
    return img

def save_to_webdataset(num_samples=1000, output_path="./data/dummy_dataset.tar"):
    with tarfile.open(output_path, "w") as tar:
        for i in range(1, num_samples + 1):
            key = f"{i:06d}"
            caption = f"This is a dummy caption for image {i}."
            image = create_dummy_image(f"Image {i}")

            # Save image to bytes
            img_buf = io.BytesIO()
            image.save(img_buf, format="JPEG")
            img_buf.seek(0)
            
            # Save caption to bytes
            txt_buf = io.BytesIO(caption.encode("utf-8"))
            txt_buf.seek(0)
            
            # Add image
            img_info = tarfile.TarInfo(name=f"{key}.jpg")
            img_info.size = len(img_buf.getbuffer())
            tar.addfile(img_info, img_buf)
            
            # Add caption
            txt_info = tarfile.TarInfo(name=f"{key}.txt")
            txt_info.size = len(txt_buf.getbuffer())
            tar.addfile(txt_info, txt_buf)
            
    print(f"Created dummy dataset: {output_path}")

if __name__ == "__main__":
    save_to_webdataset(num_samples=1000, output_path="dummy_dataset.tar")
