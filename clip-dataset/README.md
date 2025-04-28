The command to download the pd12m dataset from the parquet file.
```bash
img2dataset --url_list "./split/chunk_1.parquet" --input_format "parquet" --url_col "url" --caption_col "caption" --output_format webdataset --number_sample_per_shard=5000 --skip_reencode=True --output_folder "./data" --processes_count 16 --thread_count 64 --resize_mode no
```   

