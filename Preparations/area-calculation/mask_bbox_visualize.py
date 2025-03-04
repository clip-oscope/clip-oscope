from PIL import Image
from lang_sam import LangSAM
import torchvision
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import json
from lang_sam import utils


def get_info(result):
    unique_objects=list(set(result["labels"]))
    labels_series=pd.Series(result["labels"])
    box_list=[]
    area_list=[]
    mask_area_list=[]
    label_list=[]
    for obj in unique_objects:
        idx_list=labels_series==obj
        obj_boxes=result["boxes"][idx_list]
        obj_masks=result["masks"][idx_list]
        areas=torchvision.ops.box_area(torch.from_numpy(obj_boxes))
        max_arg=areas.argmax()
        max_area=areas.max()
        max_box=obj_boxes[max_arg]
        max_mask_area=obj_masks[max_arg].sum()
        box_list.append(max_box)
        area_list.append(max_area.item())
        mask_area_list.append(max_mask_area.item())
        label_list.append(obj)
    return box_list,area_list,mask_area_list,label_list

def update_dict(result_dict,image_ids,objects,results):
    for i in range(len(results)):
        if len(results[i]["labels"])==0:
            continue
        new_boxes,new_areas,new_mask_areas,new_labels=get_info(results[i])
        result_dict[image_ids[i]]={"objects":"","found_objects":{}}
        result_dict[image_ids[i]]["objects"]=objects[i]
        for j,label in enumerate(new_labels):
            if label not in result_dict[image_ids[i]]["found_objects"].keys():
                result_dict[image_ids[i]]["found_objects"][label]={"coords":None,"area":None,"mask_area":None}
                
            result_dict[image_ids[i]]["found_objects"][label]["coords"]=new_boxes[j].tolist()
            result_dict[image_ids[i]]["found_objects"][label]["area"]=new_areas[j]
            result_dict[image_ids[i]]["found_objects"][label]["mask_area"]=new_mask_areas[j]       



def custom_forward(model,images_pil,texts_prompt,box_threshold= 0.3,text_threshold = 0.25):
    for i, prompt in enumerate(texts_prompt):
            if prompt[-1] != ".":
                texts_prompt[i] += "."
    inputs = model.gdino.processor(images=images_pil, text=texts_prompt,padding=True, return_tensors="pt").to(model.gdino.model.device)
    with torch.no_grad():
        outputs = model.gdino.model(**inputs)

        results = model.gdino.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[k.size[::-1] for k in images_pil],
        )
    gdino_results=results


    all_results = []
    sam_images = []
    sam_boxes = []
    sam_indices = []
    for idx, result in enumerate(gdino_results):
        result = {k: (v.cpu().numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}
        processed_result = {
            **result,
            "masks": [],
            "mask_scores": [],
        }

        if result["labels"]:
            sam_images.append(np.asarray(images_pil[idx]))
            sam_boxes.append(processed_result["boxes"])
            sam_indices.append(idx)

        all_results.append(processed_result)
    if sam_images:
        masks, mask_scores, _ = model.sam.predict_batch(sam_images, xyxy=sam_boxes)
        for idx, mask, score in zip(sam_indices, masks, mask_scores):
            all_results[idx].update(
                {
                    "masks": mask,
                    "mask_scores": score,
                }
            )
        
    return all_results

def main(args):
    # Initialize the LangSAM model
    model = LangSAM()
    
    # Load and process the image
    image_pil = Image.open(args.image_path).convert("RGB")
    
    # Use provided text prompt if available, otherwise extract from filename
    if args.text_prompt:
        text_prompt = args.text_prompt
    else:
        # Create text prompt from image name
        img_name = os.path.basename(args.image_path)
        img_id = img_name.split(".")[0]
        objects = img_id.split("_")
        text_prompt = " . ".join(objects)
    
    # Process the image
    results = custom_forward(model, [image_pil], [text_prompt])
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize and save results
    output_image = utils.draw_image(image_pil, 
                                  results[0]["masks"], 
                                  results[0]["boxes"], 
                                  results[0]["scores"], 
                                  results[0]["labels"])
    
    # Save the visualization
    output_path = os.path.join(args.output_dir, f"{img_id}_visualized.png")
    output_image.save(output_path)
    
    # Save the detection results
    result_dict = {}
    result_dict[img_id] = {
        "objects": text_prompt,
        "found_objects": {}
    }
    
    # Get detection info using the existing get_info function
    if len(results[0]["labels"]) > 0:
        boxes, areas, mask_areas, labels = get_info(results[0])
        for j, label in enumerate(labels):
            result_dict[img_id]["found_objects"][label] = {
                "coords": boxes[j].tolist(),
                "area": areas[j],
                "mask_area": mask_areas[j]
            }
    
    # Save results to JSON file
    json_output_path = os.path.join(args.output_dir, f"{img_id}_results.json")
    with open(json_output_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    
    print(f"Image saved to {output_path}")
    print(f"Detection results saved to {json_output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Object detection and segmentation with LangSAM')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the results')
    parser.add_argument('--text_prompt', type=str, default=None,
                        help='Text prompt for object detection. If not provided, will be extracted from image filename.')
    parser.add_argument('--cuda_device', type=int, default=None,
                        help='CUDA device ID to use (e.g., 0, 1, etc.). If not set, uses all available devices.')
    
    args = parser.parse_args()
    main(args)
