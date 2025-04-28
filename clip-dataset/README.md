The command to download the pd12m dataset from the parquet file.
```bash
img2dataset --url_list "<parquet file path>" --input_format "parquet" --url_col "url" --caption_col "caption" --output_format webdataset --number_sample_per_shard=5000 --skip_reencode=True --output_folder "<output_folder>" --processes_count 16 --thread_count 64 --resize_mode no
```   

The commands to download the pd12m dataset directly from huggingface-cli
```bash
mkdir data
pip install huggingface_hub
chmod +x download.sh 
./download.sh <YOUR_DIRECTORY>
```

Create the Necessaray folders
```bash
mkdir imagenet
cd imagenet
mkdir tmp-val
```

Fetch the Validation Data
```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
```

Uncompress the validation data
```bash
tar -xvf ILSVRC2012_img_val.tar -C ./tmp-val
```