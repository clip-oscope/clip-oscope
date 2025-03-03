# import libraries

import torch
from PIL import Image
import open_clip
import numpy as np
import os
from tqdm.auto import tqdm
import os


# dataset paths for one_object and custom dataset. 
one_object_dataset_path = 'CoCo-OneObject'
many_objects_dataset_path = {'CoCo-FourObject-Middle-Big': 4} # path: number of objects


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# function for loading clip models
def load_model(model_name='ViT-B-32', pretrained='openai'):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    return model, preprocess


# Extract Image Features function
def extract_image_features(paths, model, preprocess):
    inputs = torch.cat([preprocess(Image.open(path).convert('RGB')).unsqueeze(0) for path in paths], dim=0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        features = model.encode_image(inputs).float()
        features /= features.norm(dim=-1, keepdim=True)
    return features


# different metrics to report

def calculate_metrics(actual, predicted):
    relevant_set = set(actual)
    retrieved_set = set(predicted)

    true_positives = len(relevant_set & retrieved_set)
    false_positives = len(retrieved_set - relevant_set)
    false_negatives = len(relevant_set - retrieved_set)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def average_precision(actual, predicted):
    relevant_set = set(actual)
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in relevant_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / len(relevant_set) if relevant_set else 0.0

def dcg(relevance_scores):
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

def ndcg(actual, predicted):
    relevance_scores = [1 if doc in actual else 0 for doc in predicted]
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)

    dcg_score = dcg(relevance_scores)
    idcg_score = dcg(ideal_relevance_scores)

    return dcg_score / idcg_score if idcg_score > 0 else 0

def mean_average_precision(actual_list, predicted_list):
    ap_scores = [average_precision(a, p) for a, p in zip(actual_list, predicted_list)]
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0

def mean_ndcg(actual_list, predicted_list):
    ndcg_scores = [ndcg(a, p) for a, p in zip(actual_list, predicted_list)]
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

def precision_recall_at_n(actual, predicted, n):
    relevant_set = set(actual)
    retrieved_set = set(predicted[:n])

    true_positives = len(relevant_set & retrieved_set)

    precision_at_n = true_positives / n if n > 0 else 0
    recall_at_n = true_positives / len(relevant_set) if relevant_set else 0

    return precision_at_n, recall_at_n


# Find Top K

def top_k(dataset_path, n, model, preprocess, one_objects_features):
    topn_correct = 0
    object_correct_in_topn = [0] * n
    batch_size = 8

    actual = list()
    predicted = list()

    base_images_addresses = [image_name for image_name in os.listdir(f'./{dataset_path}')]
    n_objects_most_similarity = [0] * n

    for i in tqdm(range(0, len(base_images_addresses), batch_size), desc="Processing image"):
        batch_images = base_images_addresses[i:i + batch_size]
        base_image_features = extract_image_features([f'./{dataset_path}/{batch_image}' for batch_image in batch_images], model, preprocess).to(device)
        cos_similarities = (base_image_features @ one_objects_features.T)
        cos_similarities = cos_similarities.cpu().numpy()
        for j, base_image_address in enumerate(batch_images):
            image_cos_similarities = cos_similarities[j]
            sorted_indices = np.argsort(image_cos_similarities)
            topn_indices = sorted_indices[-n:]
            topn_objects = [one_objects_classes[k] for k in topn_indices]
            objects_in_image = base_image_address[:-4].split('_')
            actual.append(objects_in_image)
            predicted.append(topn_objects)
            actual_indices = np.array([one_objects_classes.index(object_i) for object_i in objects_in_image])
            
            if all(ob in objects_in_image for ob in topn_objects):
                topn_correct += 1
                most_similar_object_index = np.argmax(image_cos_similarities)
                most_similar_object = one_objects_classes[most_similar_object_index]

                for k in range(n):
                    if most_similar_object == objects_in_image[k]:
                        object_correct_in_topn[k] += 1
                        break
            most_value_index = max(range(n), key=lambda j: list(sorted_indices).index(actual_indices[j]))
            n_objects_most_similarity[most_value_index] += 1
    return base_images_addresses, topn_correct, object_correct_in_topn, actual, predicted, n_objects_most_similarity


# IOR function to report IOR results
def report_ior(dataset_path, n, model_name, model, preprocess, one_objects_features):
    
    base_images_addresses, topn_correct, object_correct_in_topn, actual, predicted, n_objects_most_similarity = top_k(dataset_path, n, model, preprocess, one_objects_features)

    topn_accuracy = topn_correct / len(base_images_addresses)
    object_accuracies_in_topn = [correct / topn_correct if topn_correct else 0 for correct in object_correct_in_topn]

    print("IOR Results:")

    print(f'{dataset_path}, {model_name}')
    print(f'Top-{n} Accuracy: {topn_accuracy:.4f}')
    for i, accuracy in enumerate(object_accuracies_in_topn):
        print(f'Object {i+1} Accuracy in Correct Top-{n}: {accuracy:.4f}')
    print()
    for i in range(len(n_objects_most_similarity)):
        print(f'Object {i + 1} has most similarity between {n} objects: {n_objects_most_similarity[i] / len(base_images_addresses)}')
    precisions, recalls, f1_scores = [], [], []
    p_at_n, r_at_n = [], []

    max_len = max(max(len(a), len(p)) for a, p in zip(actual, predicted))
    for a, p in zip(actual, predicted):
        precision, recall, f1 = calculate_metrics(a, p)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        p_at_n_query, r_at_n_query = [], []
        for n in range(1, max_len + 1):
            p_n, r_n = precision_recall_at_n(a, p, n)
            p_at_n_query.append(p_n)
            r_at_n_query.append(r_n)

        p_at_n.append(p_at_n_query)
        r_at_n.append(r_at_n_query)

    map_score = mean_average_precision(actual, predicted)

    mean_ndcg_score = mean_ndcg(actual, predicted)
        
    print("\nOverall Metrics:")
    print(f"Mean Precision: {sum(precisions) / len(precisions):.2f}")
    print(f"Mean Recall: {sum(recalls) / len(recalls):.2f}")
    print(f"Mean F1 Score: {sum(f1_scores) / len(f1_scores):.2f}")
    print(f"Mean Average Precision (MAP): {map_score:.2f}")
    print(f"Mean NDCG: {mean_ndcg_score:.2f}")


    final_p_at_n = list()
    final_r_at_n = list()

    for _ in range(max_len):
        final_p_at_n.append([])
        final_r_at_n.append([])

    for idx, (p_n_query, r_n_query) in enumerate(zip(p_at_n, r_at_n)):
        for n in range(max_len):
            final_p_at_n[n].append(p_n_query[n])
            final_r_at_n[n].append(r_n_query[n])

    for i in range(max_len):
        print(f"Mean P@{i+1}: {sum(final_p_at_n[i]) / len(final_p_at_n[i]):.2f}, Mean R@{i+1}: {sum(final_r_at_n[i]) / len(final_r_at_n[i]):.2f}")
    print(50 * '-')



# one object names
one_objects_classes = [image_name.replace('.png', '') for image_name in os.listdir(f'./{one_object_dataset_path}')]


# Experiments Setups
model_name = 'ViT-B-32'
pretrained = 'openai'

# Experiments
model, preprocess,  = load_model(model_name=model_name, pretrained=pretrained)
one_objects_features = extract_image_features([f'./{one_object_dataset_path}/{one_objects_class}.png' for one_objects_class in one_objects_classes], model, preprocess)
model_name = f'{model_name} - {pretrained}'


for dataset_path in many_objects_dataset_path.keys():
    n = many_objects_dataset_path[dataset_path]
    report_ior(dataset_path, n, model_name, model, preprocess, one_objects_features)