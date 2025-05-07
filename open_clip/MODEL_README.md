Command to install all the required dependencies.
```bash
pip install -r open_clip/requirements-training.txt
pip install open_clip_torch
```

Command to login into the Weights and Biases.
```bash
wandb login
```

Command to train the model
```bash
cd open_clip/src
python -m open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --wandb-project-name swin-open-clip \
    --train-data '/teamspace/studios/this_studio/llava-cvt-swin/clip-dataset/data/{00155..00174}.tar \
    --train-num-samples 99911 \
    --val-data '/teamspace/studios/this_studio/llava-cvt-swin/clip-dataset/data/{00175..00176}.tar' \
    --val-num-samples 9992 \
    --dataset-type webdataset \
    --imagenet-val=/teamspace/studios/this_studio/llava-cvt-swin/clip-dataset/imagenet/val \
    --name 'pd12m-100k-swin-roberta-1-e-3-cosine-grad-accum-4-256' \
    --warmup 60 \
    --lr-scheduler "cosine" \
    --batch-size 256 \
    --accum-freq 4 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=3 \
    --precision amp_bf16\
    --workers=20 \
    --log-every-n-steps 4 \
    --seed 42 \
    --logs /teamspace/studios/this_studio/llava-cvt-swin/open_clip/logs \
    --model swinv2_base_window12_192_roberta
```

Login to your HuggingFace using this command by first getting the personal access token
with read and write permissions to the repository.
```bash
huggingface-cli login
```

Command to push the model to the HuggingFace
```bash
python -m open_clip.push_to_hf_hub --model swinv2_base_window12_192_roberta --pretrained /teamspace/studios/this_studio/llava-cvt-swin/open_clip/logs/pd12m-100k-swin-roberta-5-e-4-grad-accum-2-256/checkpoints/epoch_3.pt --repo-id SuryaKrishna02/swinv2-roberta-openclip
```