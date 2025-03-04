# Preparations

This directory contains various components and tools for working with text and object detection, language models, and attention mechanisms. Below you'll find information about each component and how to use them.

## Project Structure

- `open-clip-attention/`: Implementation of attention mechanisms using OpenCLIP for generating bounding box of objects and their masks.
- `Text Object Detection/`: Tools for identifying objects in text using LLaMA 3
- `Lang-Sam/`: Language-guided Segment Anything Model implementation
- `Reorder Text Objects/`: Tools for reordering detected text objects
- `Text Object Replacement/`: Tools for replacing detected text objects




## Usage Instructions

### Area-Calculation

### Requirements
1. [Language Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything)

## Prerequisites

- Python 3.11 or higher

## Installation

### Installing `lang-sam`
You can follow the detailed instructions in their repositories to install [Lang-SAM](https://github.com/luca-medeiros/lang-segment-anything/tree/main#installation) and [SAM](https://github.com/facebookresearch/sam2#installation). Below is a brief guide for installation:

1. Create a Conda Environment

Before installing, we recommend using a virtual environment:
```bash
conda create -n lsam python=3.11
conda activate lsam
```

2. Install PyTorch Dependencies

Install the required PyTorch dependencies:
```bash
pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu124
```
3. Install CUDA Toolkit (if necessary)

Before installing `lang-sam`, you'll need `nvcc` for compilation, as mentioned in the SAM documentation.
Ensure you install the CUDA Toolkit version that matches your PyTorch CUDA version. You can find the appropriate version [here](https://developer.nvidia.com/cuda-toolkit-archive).

4. Set CUDA_HOME

Find where `nvcc` is installed and set `CUDA_HOME`:

```bash
sudo find / -name nvcc  # Find the CUDA compiler location
export CUDA_HOME=/usr/local/cuda-12.4/
```
5. Install `lang-sam`:

Finally, install `lang-sam`:
```bash
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
```


### Usage

The `area-calculation` module computes the areas of detected and segmented objects in images using Lang-SAM.


1. Navigate to the project directory:
```bash
cd area-calculation
```

2. Run `mask_bbox.py` with the required arguments:
```bash
python mask_bbox.py --data_root /path/to/data --output_dir /path/to/output
```
Or run `mask_bbox_visualize.py` for a single image and visulization
```bash
python mask_bbox_visualize.py --image_path ./assets/banana_horse_sheep.png --output_dir ./assets
```

### Detection Results Format

The output file contains object detection and segmentation results with the following structure:

### File Structure

```json
{
    "image_name.png": {
        "objects": "object1 . object2 . object3.",
        "found_objects": {
            "object1": {
                "coords": [x1, y1, x2, y2],
                "area": float,
                "mask_area": float
            },
            ...
        }
    }
}
```

### Fields Explanation

### Top Level
- Each image is represented by a key-value pair where the key is the image filename
- Each image entry contains two main fields: `objects` and `found_objects`

### Objects Field
- `objects`: A string containing all expected objects in the image, separated by " . " 
  - Example: "airplane . plate . axe."

### Found Objects Field
- `found_objects`: Contains detection results for each object found in the image
  - Each object has the following properties:
    - `coords`: Array of 4 values [x1, y1, x2, y2] representing the bounding box coordinates
      - x1, y1: Top-left corner coordinates
      - x2, y2: Bottom-right corner coordinates
    - `area`: The area of the bounding box in pixels
    - `mask_area`: The area of the object's segmentation mask in pixels

### Example Entry

```json
{
    "airplane_plate_axe.png": {
        "objects": "airplane . plate . axe.",
        "found_objects": {
            "plate": {
                "coords": [244.02, 197.65, 313.72, 267.41],
                "area": 4861.78,
                "mask_area": 3730.0
            }
        }
    }
}
```

Note: Not all objects listed in the `objects` field may have corresponding entries in `found_objects` if they were not successfully detected.


### LLaMA 3 Core Implementation

The `llama3` directory contains the core LLaMA 3 model implementation:

1. Navigate to the directory:
```bash
cd llama3
```

2. The main implementation is in `llama3.ipynb`:
   - This notebook contains the core LLaMA 3 model setup and usage
   - Make sure you have the required model weights (see Prerequisites section)
   - Use Jupyter to open and run the notebook

### Identifying Objects in Text

To identify objects in text using LLaMA 3:

1. Navigate to the directory:
```bash
cd Identifing\ Objects\ in\ Text\ via\ llama3
```

2. Follow the component's setup instructions

### Lang-SAM Integration

For language-guided object segmentation:

1. Navigate to the directory:
```bash
cd Lang-Sam
```

2. Follow the setup instructions in the component's README

### Text Object Operations

For reordering and replacing text objects:

1. Navigate to either:
```bash
cd Reorder\ Text\ Objects
# or
cd Text\ Object\ Replacement
```

2. Follow the specific component's instructions

## Contributing

Feel free to contribute to any of these components by:
1. Creating issues for bugs or feature requests
2. Submitting pull requests with improvements
3. Adding documentation or examples

## License

Please refer to individual component directories for specific licensing information.

## Contact

For questions or support, please create an issue in the respective component's directory.
