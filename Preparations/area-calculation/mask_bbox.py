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

class CustomDataset(Dataset):
    def __init__(self, root):
        super(CustomDataset, self).__init__()
        self.root=root
        self.file_list = sorted(os.listdir(root))
        self.prompt_list=[]
        for file in self.file_list:
          img_id=file.split(".")[0]
          objects=img_id.split("_")
          text_prompt = " . ".join(objects)
          self.prompt_list.append(text_prompt)
    def __len__(self):
        return len(self.prompt_list)
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slice indexing
            files = self.file_list[idx]
            prompts = self.prompt_list[idx]
            images = []
            for f in files:
                img = Image.open(os.path.join(self.root,f)).convert("RGB")
                images.append(img)
            return files, images, prompts
        else:
            # Handle single integer indexing
            img, prompt = self.file_list[idx], self.prompt_list[idx]
            image_pil = Image.open(os.path.join(self.root,img)).convert("RGB")
            return img, image_pil, prompt

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
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        
    model = LangSAM()
    dataset = CustomDataset(args.data_root)
    result_dict = {}
    
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch_slice = slice(i, min(i + args.batch_size, len(dataset)))
        id_list, image_list, prompt_list = dataset[batch_slice]
        results = custom_forward(model, image_list, prompt_list)
        update_dict(result_dict, id_list, prompt_list, results)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results as JSON
    output_path = os.path.join(args.output_dir, 'detection_results.json')
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Object detection and segmentation with LangSAM')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing the images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing (default: 16)')
    parser.add_argument('--cuda_device', type=int, default=None,
                        help='CUDA device ID to use (e.g., 0, 1, etc.). If not set, uses all available devices.')
    
    args = parser.parse_args()
    main(args)
