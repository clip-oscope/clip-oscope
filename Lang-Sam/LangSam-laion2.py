from torchvision import datasets
from torchvision import transforms
import groundingdino.datasets.transforms as T
import torchvision
import os
from torch.utils.data import DataLoader
from lang_sam import LangSAM
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import lang_sam.utils as utils
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide


model = LangSAM()

H,W=600,800

root="laion_subset"
files=sorted(os.listdir(root))
with open("result_laion.json","r") as f:
    json_dict=json.load(f)


transform_sam = ResizeLongestSide(model.sam.model.image_encoder.img_size)



new_dict={}
files_count=len(files)
batch_size=4
save_idx=0
for index in tqdm(range(0,files_count,batch_size)):
    save_idx+=1
    image_list=[]
    key_list=[]
    if index+batch_size>files_count:
        batch_size=(index+batch_size)-files_count
    for i in range(batch_size):
        img_idx=index+i
        image_pil=Image.open(root+"/"+files[img_idx])
        image_pil=image_pil.resize((W,H)).convert('RGB')
        key=files[img_idx]
        if key in json_dict.keys():
            key_list.append(key)
            image_list.append(transform_sam.apply_image(np.asarray(image_pil)))
    input_image_torch_list=[]
    for i in range(len(image_list)):
        input_image = image_list[i]
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image_torch_list.append(input_image_torch)
    input_image_torch=torch.cat(input_image_torch_list,dim=0)
    model.sam.set_torch_image(input_image_torch.cuda(), [H,W])
    for idx,key in enumerate(key_list):
        new_dict[key]={}
        new_dict[key]["caption"]=json_dict[key]["caption"]
        new_dict[key]["objects"]=json_dict[key]["objects"]
        new_dict[key]["found_objects"]={}
        for object in json_dict[key]["found_objects"].keys():
            box=torch.tensor(json_dict[key]["found_objects"][object]["coords"])
            box_area=json_dict[key]["found_objects"][object]["area"]

            transformed_boxes = model.sam.transform.apply_boxes_torch(box, [H,W])
            sparse_embeddings, dense_embeddings = model.sam.model.prompt_encoder(
                        points=None,
                        boxes=transformed_boxes.cuda(),
                        masks=None,
                    )
            low_res_masks, iou_predictions = model.sam.model.mask_decoder(
                        image_embeddings=model.sam.features[idx][None],
                        image_pe=model.sam.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
            masks = model.sam.model.postprocess_masks(low_res_masks, model.sam.input_size, model.sam.original_size)
            masks = masks > model.sam.model.mask_threshold
            masks = masks.squeeze(1).cpu()
            #masks_list.append(masks)


            #masks=model.predict_sam(image_pil,box)
            #masks = masks.squeeze(1)
            if len(masks)>0:
                mask_area=masks.sum()
                new_dict[key]["found_objects"][object]={"coords":box.tolist(),"box_area":box_area,"mask_area":mask_area.item()}
    if save_idx%10000==0:
        with open(f"new_results_laion_{index}.json","w") as f:
            r=json.dumps(new_dict)
            f.write(r)