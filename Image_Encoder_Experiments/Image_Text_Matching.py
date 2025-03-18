import argparse
from tqdm.auto import tqdm
import os
import pandas as pd
import torch
from PIL import Image
import open_clip

# Command-line argument parser
parser = argparse.ArgumentParser(description='Image-Text Matching Experiment')
parser.add_argument('--one_object_dataset_path', type=str, default='CoCo-OneObject', help='Path to one-object dataset')
parser.add_argument('--many_objects_dataset_path', type=str, default='CoCo-FourObject-Middle-Big:4:2', help='Path:objects_size:biased_object_pos for many-objects dataset')
parser.add_argument('--model_name', type=str, default='ViT-B-32', help='CLIP model name')
parser.add_argument('--pretrained', type=str, default='openai', help='Pretrained model source')
parser.add_argument('--text_batch_size', type=int, default=256, help='Batch size for text embedding')
parser.add_argument('--image_batch_size', type=int, default=64, help='Batch size for image embedding')
args = parser.parse_args()

# Parse dataset paths
many_objects_dataset_path = {}
dataset_parts = args.many_objects_dataset_path.split(',')
for part in dataset_parts:
    path, size, pos = part.split(':')
    many_objects_dataset_path[path] = (int(size), int(pos))

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function for loading clip models
def load_model(model_name='ViT-B-32', pretrained='openai'):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


# Extract Image Features function

def extract_image_features(model, preprocess, paths):
    inputs = torch.cat([preprocess(Image.open(path).convert('RGB')).unsqueeze(0) for path in paths], dim=0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
      features = model.encode_image(inputs).float()
      features /= features.norm(dim=-1, keepdim=True)
    return features

def get_dict_image_embedings(model, preprocess, dataset_path, image_batch_size = 64):
    listdir = os.listdir(f'./{dataset_path}')
    image_embedings = {}
    for i in tqdm(range(0, len(listdir), image_batch_size), desc="Dict Image Embedings"):
        batch_images_name = listdir[i:i + image_batch_size]
        image_features = extract_image_features(model, preprocess, [f'./{dataset_path}/{image_name}' for image_name in batch_images_name])
        for i, image_name in enumerate(batch_images_name):
            image_embedings[image_name] = image_features[i].unsqueeze(0)
    return image_embedings


# Extract Text Features function

def extract_text_features(model, tokenizer, texts):
    tokenizde_propmts = tokenizer(texts).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        features = model.encode_text(tokenizde_propmts).float()
        features /= features.norm(dim=-1, keepdim=True)
    return features

def get_dict_text_embedings(model, tokenizer, texts, text_batch_size = 256):
    text_embedings = {}
    for i in tqdm(range(0, len(texts), text_batch_size), desc="Dict Text Embedings"):
        batch_texts = texts[i:i + text_batch_size]
        text_features = extract_text_features(model, tokenizer, batch_texts)
        for i, text in enumerate(batch_texts):
            text_embedings[text] = text_features[i].unsqueeze(0)
    return text_embedings


# Functions to change the postions ob object in text

def move_to_position(lst, current_index, target_index):
    element = lst.pop(current_index)
    lst.insert(target_index, element)
    return lst

def swap_list(lst, idx1, idx2):
    lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
    return lst

def join_with_and(lst):
    return ' and '.join(lst[:-1]) + ' and ' + lst[-1]

# Prepare wrong and correct dataset
def prepare_dataset(one_objects_dataset, biased_object_index, dataset_path, is_first_scenario):
    correct_texts = []
    wrong_texts = []
    image_names = []
    
    listdir = os.listdir(f'./{dataset_path}')
    for I in listdir:
        image_objects = I.replace('.png', '').split('_')
        if is_first_scenario:
            correct_text = join_with_and(swap_list(image_objects.copy(), biased_object_index, 0))
        else:
            correct_text = join_with_and(swap_list(image_objects.copy(), biased_object_index, len(image_objects) - 1))
        changed_position_objects = swap_list(image_objects.copy(), biased_object_index, 0)
        
        
        for j in range(len(one_objects_dataset)):
            if one_objects_dataset[j] in image_objects:
                continue
            if is_first_scenario:
                changed_position_objects = changed_position_objects[1:]
                changed_position_objects = [one_objects_dataset[j]] + changed_position_objects
            else:
                changed_position_objects = changed_position_objects[:-1]
                changed_position_objects = changed_position_objects + [one_objects_dataset[j]]
            changed_last_object_text = join_with_and(changed_position_objects)
            wrong_texts.append(changed_last_object_text)
            correct_texts.append(correct_text)
            image_names.append(I)

    df = pd.DataFrame({'image_name': image_names, 'correct_text': correct_texts, 'wrong_text': wrong_texts})
    return df



# Count corrects

def calculate_correct(model, tokenizer, image_embedings, correct_texts, wrong_texts_embeding):
    correct_texts_features =  extract_text_features(model, tokenizer, correct_texts)

    correct_texts_probs = torch.diagonal(image_embedings @ correct_texts_features.T)
    wrong_texts_probs = torch.diagonal(image_embedings @ wrong_texts_embeding.T)
    return (correct_texts_probs > wrong_texts_probs).sum().item()


# Image Text Matching function

def image_text_matching(model, tokenizer, model_name_full, dataset_path, df_dataset, image_embedings, wrong_texts_embeding, batch_size, is_first_scenario):
    
    
    correct = 0
    for i in tqdm(range(0, len(df_dataset), batch_size), desc="Image Text Matching"):
        images_name = df_dataset['image_name'][i:i + batch_size].tolist()
        correct_texts = df_dataset['correct_text'][i:i + batch_size].tolist()
        wrong_texts = df_dataset['wrong_text'][i:i + batch_size].tolist()
        correct += calculate_correct(model=model, tokenizer=tokenizer, 
                                        image_embedings=torch.cat([image_embedings[image_name] for image_name in images_name], dim=0).to(device), 
                                        correct_texts=correct_texts, 
                                        wrong_texts_embeding=torch.cat([wrong_texts_embeding[text] for text in wrong_texts], dim=0).to(device))
    print(f'Model: {model_name_full} // Dataset: {dataset_path}')
    if is_first_scenario:
        print(f'First Scenario: Acc = {correct / len(df_dataset)}') 
    else:
        print(f'Second Scenario: Acc = {correct / len(df_dataset)}') 
    print(50 * '-')
        

# Experiments
one_objects_dataset = [I.replace('.png', '') for I in os.listdir(f'./{args.one_object_dataset_path}')]
model, preprocess, tokenizer  = load_model(model_name=args.model_name, pretrained=args.pretrained)
model_name_full = f'{args.model_name} - {args.pretrained}'

for dataset_path in many_objects_dataset_path.keys():
    n, biased_object_index = many_objects_dataset_path[dataset_path]

    print('Get Image Embedings:')
    image_embedings = get_dict_image_embedings(model=model, preprocess=preprocess, dataset_path=dataset_path, image_batch_size=args.image_batch_size)

    df_dataset = prepare_dataset(one_objects_dataset=one_objects_dataset, biased_object_index=biased_object_index - 1, dataset_path=dataset_path, is_first_scenario=True)
    wrong_texts_embeding = get_dict_text_embedings(model=model, tokenizer=tokenizer, texts=df_dataset['wrong_text'], text_batch_size=args.text_batch_size)
    image_text_matching(model=model, tokenizer=tokenizer, model_name_full=model_name_full, dataset_path=dataset_path, df_dataset=df_dataset, image_embedings=image_embedings, wrong_texts_embeding=wrong_texts_embeding, batch_size=args.text_batch_size, is_first_scenario=True)

    df_dataset = prepare_dataset(one_objects_dataset=one_objects_dataset, biased_object_index=biased_object_index - 1, dataset_path=dataset_path, is_first_scenario=False)
    wrong_texts_embeding = get_dict_text_embedings(model=model, tokenizer=tokenizer, texts=df_dataset['wrong_text'], text_batch_size=args.text_batch_size)
    image_text_matching(model=model, tokenizer=tokenizer, model_name_full=model_name_full, dataset_path=dataset_path, df_dataset=df_dataset, image_embedings=image_embedings, wrong_texts_embeding=wrong_texts_embeding, batch_size=args.text_batch_size, is_first_scenario=False)