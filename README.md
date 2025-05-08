# SwinV2-LLaVA Experiment

This repository contains the code for the paper "The SwinV2-LLaVA Experiment: A Practical Guide to Avoid Integration Pitfalls". This README provides detailed instructions for implementing and reproducing our experimental setup for training a SwinV2-based vision encoder and integrating it into the LLaVA framework.

## Overview

Our experiment consists of two main phases:
1. Training a SwinV2-based image encoder using the OpenCLIP framework
2. Integrating this encoder into the LLaVA architecture

This guide documents the complete process, including solutions to common challenges and bottlenecks encountered during implementation.

## 1. Vision Encoder Training with SwinV2

### 1.1 Dataset Preparation

The dataset preparation is a critical step that often presents significant challenges. We provide two alternative methods to download the PD12M dataset.

#### Change to the clip-dataset directory
```bash
cd clip-dataset
```

#### Method 1: Download from Parquet File (Less Reliable)
```bash
img2dataset --url_list "<parquet file path>" --input_format "parquet" --url_col "url" --caption_col "caption" --output_format webdataset --number_sample_per_shard=5000 --skip_reencode=True --output_folder "<output_folder>" --processes_count 16 --thread_count 64 --resize_mode no
```

#### Method 2: Download Directly from HuggingFace (Recommended)
```bash
mkdir data
pip install huggingface_hub
chmod +x download.sh 
./download.sh <YOUR_DIRECTORY>
```

#### Validate Dataset Size
Use the following command to calculate the number of images in the downloaded .tar files:
```bash
python pd12m_image_stats.py "<DATA_FOLDER>" --exclude <VAL_FILE_NAME_1> <VAL_FILE_NAME_2>
```

Example output:
```
Total images (including excluded files): 109903
Total excluded images: 9992
Final image count (after exclusions): 99911
```

### 1.2 ImageNet Validation Setup

For evaluation purposes, we use the ImageNet validation set. Follow these steps to prepare it:

```bash
mkdir imagenet
cd imagenet
mkdir tmp-val
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xvf ILSVRC2012_img_val.tar -C ./tmp-val
```

Rearrange the validation data for compatibility with ImageNet validation:
```bash
chmod +x val_prep.sh 
./val_prep.sh
```

### 1.3 SwinV2 Vision Encoder Training

#### Install Dependencies

First, install all the required dependencies:
```bash
pip install -r open_clip/requirements.txt
pip install open_clip_torch
```

#### Configure Experiment Tracking

Login to Weights and Biases for experiment tracking:
```bash
wandb login
```

#### Model Configuration

The model configuration should be modified in `open_clip/src/open_clip/model_configs` to include the SwinV2 architecture specifications.

#### Training Command

To train the SwinV2 vision encoder using OpenCLIP, use the following command:

```bash
cd open_clip/src
python -m open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --wandb-project-name swin-open-clip \
    --train-data '<TRAIN_DATA_PATH>/{00155..00174}.tar \
    --train-num-samples <NUM_TRAIN_SAMPLES> \
    --val-data '<VAL_DATA_PATH>/data/{00175..00176}.tar' \
    --val-num-samples <NUM_VAL_SAMPLES> \
    --dataset-type webdataset \
    --imagenet-val=<IMAGENET_VAL_PATH> \
    --name 'pd12m-100k-swin-roberta-1-e-3-cosine-grad-accum-4-256' \
    --warmup 60 \
    --batch-size 256 \
    --accum-freq 4 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=3 \
    --precision amp_bf16\
    --workers=20 \
    --log-every-n-steps 4 \
    --seed 42 \
    --logs <LOGS_PATH> \
    --model <MODEL_CONFIG_NAME>
```

Key hyperparameters:
- Batch size: 256 (with gradient accumulation for effective batch size of 1024)
- Learning rate: 1e-3 with cosine scheduler and warmup
- Weight decay: 0.1
- Optimizer: AdamW
- Training duration: 3 epochs
- Mixed precision: BF16

#### Publishing the Model

Login to HuggingFace (first obtain a personal access token with read and write permissions):
```bash
huggingface-cli login
```

Push the trained model to HuggingFace:
```bash
python -m open_clip.push_to_hf_hub --model <MODEL_CONFIG_NAME> --pretrained <TRAINED_CHECKPOINT_PATH> --repo-id <HUGGINGFACE_REPO>
```

For stable training, we recommend:
- Effective batch size of at least 10,000 (use gradient accumulation if GPU memory is limited)
- Longer training schedules (10+ epochs) for better convergence
- Careful monitoring of loss curves to detect convergence issues

## 2. LLaVA Integration

### 2.1 Installation and Setup

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2.2 Modifying the OpenCLIP Encoder

The most challenging part of our experiment was properly registering the custom SwinV2 encoder as a Vision Tower within the LLaVA architecture. You need to modify the encoder implementation in `LLaVA-NeXT/llava/model/multimodal_encoder/open_clip_encoder.py`.

#### Update the Hidden Size Dictionary

First, add your SwinV2 model to the hidden size dictionary:

```python
HIDDEN_SIZE_DICT = {
    "ViT-H-14-378-quickgelu": 1280,
    # Add your SwinV2 model configuration here
    "SwinV2B-RoBERTaL": 768,  # Replace with your model's embedding dimension
}
```

#### Implement Required Methods

The integration requires implementing approximately 10 methods to ensure compatibility. We outline the key methods below, with 2-3 already implemented in the codebase and others that need to be added:

```python
# Create a custom vision tower class for SwinV2
class OpenCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower_path, freeze=True, select_layer=-1, select_feature_type="patch"):
        super().__init__()
        
        # Parse model name from vision_tower_path
        self.model_name = self._get_model_name(vision_tower_path)
        
        # Initialize the SwinV2 model from OpenCLIP
        self.model = create_model(
            self.model_name,
            pretrained=vision_tower_path,
            precision="fp16"
        )
        
        self.hidden_size = HIDDEN_SIZE_DICT.get(self.model_name)
        if self.hidden_size is None:
            raise ValueError(f"Hidden size for model {self.model_name} not found in HIDDEN_SIZE_DICT")
            
        self.select_layer = select_layer
        self.select_feature_type = select_feature_type
        self.image_processor = None
        self.feature_size = self.model.embed_dim
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def _get_model_name(self, vision_tower_path):
        # Extract model name from path or config
        # Implementation depends on your specific setup
        return "SwinV2B-RoBERTaL"  # Replace with actual logic
    
    def forward(self, images):
        # Process images through SwinV2 and return visual features
        with torch.no_grad():
            features = self.model.encode_image(images)
        return features
    
    # Method to get the feature dimension
    def get_feature_dim(self):
        return self.hidden_size
    
    # Required method to transform image for the model
    def transform_image(self, image_tensor):
        # Implement the appropriate image transformation
        # This should match the preprocessing used during training
        pass
    
    # Required method to extract features from a specific layer
    def extract_features(self, image_tensor, layer_idx=None):
        # Implement feature extraction logic
        pass
    
    # Required method to resize position embeddings if needed
    def resize_pos_embeddings(self, new_size):
        # Implement position embedding resizing for SwinV2 if necessary
        # This may differ significantly from ViT implementations
        pass
    
    # Required method to handle image batch processing
    def process_image_batch(self, image_batch):
        # Implement batch processing logic
        pass
    
    # Additional required methods based on LLaVA-NeXT implementation
    # ...
```

#### Modify the Model Registration Logic

Update the model creation and registration logic:

```python
def create_llava_model_with_swinv2(config_path, vision_tower_path):
    # Load config and modify for SwinV2
    config = AutoConfig.from_pretrained(config_path)
    config.mm_vision_tower = vision_tower_path
    config.vision_tower_type = "open_clip"  # Specify the tower type
    
    # Create model with modified config
    model = AutoModelForCausalLM.from_config(config)
    
    # Register SwinV2 vision tower
    vision_tower = OpenCLIPVisionTower(vision_tower_path)
    model.model.vision_tower = vision_tower
    
    return model
```

### 2.3 Implementation Challenges

The integration process faces several significant challenges:

1. **Architectural Differences**: SwinV2's hierarchical structure differs substantially from ViT's uniform approach, requiring careful handling of feature extraction.

2. **Incomplete Implementation**: As noted, approximately 10 methods need to be implemented in the encoder class, many of which are specific to the LLaVA architecture and not documented.

3. **Hidden Feature Dimensions**: Ensuring the correct feature dimensions are registered in `HIDDEN_SIZE_DICT` is critical for successful integration.

4. **Position Embedding Handling**: SwinV2's position encoding mechanism differs from ViT, requiring specialized implementation for resizing and processing.

5. **Feature Extraction Logic**: The LLaVA architecture expects specific feature extraction patterns that must be adapted for SwinV2's hierarchical representation.

**Important Note**: Due to these implementation challenges, the current integration is not fully functional. Researchers attempting to reproduce this work should be prepared to implement all required methods and address additional compatibility issues not documented here.

### 2.2 Dataset Preparation for LLaVA

For pretraining and instruction tuning, prepare the following datasets:

#### Download LLaVA-Pretrain dataset from HuggingFace
```bash
cd llava-dataset
mkdir data
chmod +x download.sh 
./download.sh <YOUR_DIRECTORY>
```

#### Start the pretraining on the LLaVA-Pretrain
Replace the LLM Model and Vision model with the respective models that you want to train.
```bash
cd LLaVA-NeXT
chmod +x pretrain_clip.sh
./pretrain_clip.sh
```

### 2.3 Evaluation
The following commands to used to evaluate the LLaVAOneVision-0.5B on MMMU Dataset.
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
accelerate launch lmms_eval/__main__.py \
  --model llava_onevision \
  --model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov \
  --tasks mmmu_val \
  --batch_size 1 \
  --output_path ./logs
```

## 3. Troubleshooting and Known Issues

### Training Instabilities
- If experiencing loss spikes during training, try:
  - Increasing batch size to at least 10,000
  - Lowering learning rate to 1e-5
  - Using longer warmup periods (500+ steps)

### Integration Challenges
- When registering custom encoders, ensure output feature dimensions match LLaVA's expectations
- If encountering dimension mismatches, implement a projection layer to align feature spaces
- For issues with model loading, check for any hardcoded references to ViT architectures in the LLaVA codebase

### Dataset Issues
- For broken links in PD12M, use the HuggingFace method with our download script
- If experiencing timeouts, implement exponential backoff in download scripts
- For memory issues with large datasets, use sharded loading techniques

## 4. Performance Expectations

Based on our experiments:
- Expected ImageNet accuracy with our setup: 1% top-1, 5% top-5
- Training convergence may show instability (see loss curve in paper)
- Full LLaVA integration requires careful handling of feature dimensions

## 5. Contributing

We welcome contributions to improve this codebase, particularly:
- Enhancements to dataset processing scripts
- Solutions to integration challenges