# Image Encoder Experiments

This folder contains three Python scripts for analyzing different aspects of image encoding and object understanding:

- `IOR.py` (Image-based Object Retrieval): Analyzes how well the model can retrieve specific objects from images
- `IOC.py` (Image-based Object Classification): Evaluates the model's ability to classify objects in images
- `Image_Text_Matching.py`: Implements two different scenarios of image-text matching analysis

## Installation

Install the required packages:
```bash
pip install -r requirements.txt
```

Required packages include:
- pandas
- torch
- torchvision==0.20.1+cu121
- tqdm
- pillow
- open_clip_torch==2.30.0
- matplotlib

## Dataset Input Format

All scripts require two main dataset paths as input:

1. `one_object_dataset_path`: 
   - Path to the dataset containing images with single objects
   - Used as a reference dataset for object analysis

2. `many_objects_dataset_path`:
   - Path to the dataset containing images for analysis
   - Used for performing IOR, IOC, or Image-Text Matching tasks

### Input Format Specifications

1. For Image-based Object Retrieval (IOR) and Image-based Object Classification (IOC):
```
dataset_path:number_of_objects
```
Example: `/path/to/dataset:4`. `/path/to/dataset` shows path to the many_objects_dataset and 4 show the number of objects in each image.

2. For Image-Text Matching:
```
dataset_path:number_of_objects:bias_position_to_evaluate
```
Example: `/path/to/dataset:4:2`. `/path/to/dataset` shows path to the many_objects_dataset, 4 show the number of objects in each image and 2 shows the object position for analysis.

## Usage Examples

### Common Command Line Arguments

All scripts support the following arguments:
```bash
--model_name        Model name
--pretrained        Pretrained model source
--batch_size        Batch size for processing images
```

1. Running Image-based Object Retrieval (IOR):
```bash
python IOR.py 
--one_object_dataset_path=/path/to/single_object_dataset 
--many_objects_dataset_path=/path/to/multiple_objects_dataset:4
--model_name=ViT-B-32
--pretrained=openai
--batch_size=32
```

2. Running Image-based Object Classification (IOC):
```bash
python IOC.py 
--one_object_dataset_path=/path/to/single_object_dataset 
--many_objects_dataset_path=/path/to/multiple_objects_dataset:4
--model_name=ViT-B-32
--pretrained=openai
--batch_size=32
--epochs=25        # Additional parameter specific to IOC
```

3. Running Image-Text Matching Analysis:
```bash
python Image_Text_Matching.py 
--one_object_dataset_path=/path/to/single_object_dataset 
--many_objects_dataset_path=/path/to/multiple_objects_dataset:4:2
--model_name=ViT-B-32
--pretrained=openai
--text_batch_size=128
--image_batch_size=32
```
