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
    --train-data '/teamspace/studios/this_studio/llava-cvt-swin/clip-dataset/data/00155.tar' \
    --train-num-samples 99911 \
    --val-data '/teamspace/studios/this_studio/llava-cvt-swin/clip-dataset/data/00175.tar' \
    --val-num-samples 9992 \
    --dataset-type webdataset \
    --imagenet-val=/teamspace/studios/this_studio/llava-cvt-swin/clip-dataset/imagenet/val \
    --name 'pd12m-100k-swin-roberta-5-e-4' \
    --warmup 100 \
    --batch-size 256 \
    --accum-freq 2 \
    --lr=5e-4 \
    --wd=0.1 \
    --epochs=5 \
    --precision amp_bf16\
    --workers=1 \
    --log-every-n-steps 4 \
    --seed 42 \
    --logs /teamspace/studios/this_studio/llava-cvt-swin/open_clip/logs \
    --model swin_base_patch4_window12to16_192to256_roberta
```