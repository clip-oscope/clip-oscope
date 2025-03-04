# dowload NegCLIP checkpoint: !gdown "1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ&confirm=t"
import torch
from PIL import Image
import open_clip
import numpy as np
from tqdm import tqdm
import random
from itertools import combinations

random.seed(42)


model_name = "ViT-B-32"
pretrained = "openai"
n = 2
batch_size = 512
bigger_obj_first = True


device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)


model.load_state_dict(torch.load('negCLIP.pt')['state_dict'])
model.eval()


small_objs = ["ant", "anvil", "apple", "arm", "asparagus", "axe", "banana", "bandage", "basket", "bat", "bee", "belt",
              "binoculars", "bird", "blackberry", "blueberry", "book", "boomerang", "bottlecap", "bowtie", "bracelet", "brain",
              "bread", "broccoli", "broom", "bucket", "butterfly", "cactus", "cake", "calculator", "calendar", "camera", "candle", "carrot", "cat",
              "clarinet", "clock", "compass", "cookie", "crab", "backpack", "crown", "cup", "dog", "donut", "drill",
              "duck", "dumbbell", "ear", "envelope", "eraser", "eye", "eyeglasses", "feather", "finger", "fork", "frog", "hammer",
              "hat", "headphones", "hedgehog", "helmet", "hourglass", "jacket", "keyboard", "key",
              "knife", "lantern", "laptop", "leaf", "lipstick", "lobster", "lollipop", "mailbox", "marker", "megaphone", "microphone",
              "microwave", "mosquito", "mouse", "mug", "mushroom", "necklace", "onion", "owl", "paintbrush", "parrot", "peanut",
              "pear", "peas", "pencil", "pillow", "pineapple", "pizza", "pliers", "popsicle", "postcard", "potato", "purse", "rabbit", "raccoon",
              "radio", "rake", "rhinoceros", "rifle", "sandwich", "saw", "saxophone", "scissors", "scorpion", "shoe", "shovel",
              "skateboard", "skull", "snail", "snake", "snorkel", "spider", "spoon", "squirrel", "stethoscope", "strawberry", "swan",
              "sword", "syringe", "teapot", "telephone", "toaster", "toothbrush", "trombone", "trumpet", "umbrella", "violin", "watermelon", "wheel"]

mid_objs = ["angel", "bathtub", "bear", "bed", "bench", "bicycle", "camel", "cannon", "canoe", "cello", "chair", "chandelier",
            "computer", "cooler", "couch", "cow", "crocodile", "dishwasher", "dolphin", "door", "dresser", "drums", "flamingo",
            "guitar", "horse", "kangaroo", "ladder", "mermaid", "motorbike", "panda", "penguin", "piano", "pig", "sheep", "stereo", "stove",
            "table", "television", "tiger", "zebra"]

big_objs = ["aircraft carrier", "airplane", "ambulance", "barn", "bridge", "bulldozer", "bus", "car", "castle", "church", "cloud", "cruise ship", "dragon",
            "elephant", "firetruck", "flying saucer", "giraffe", "helicopter", "hospital", "hot air balloon", "house", "moon", "mountain", "palm tree",
            "parachute", "pickup truck", "police car", "sailboat", "school bus", "skyscraper", "speedboat", "submarine", "sun", "tent", "The Eiffel Tower",
            "The Great Wall of China", "tractor", "train", "tree", "truck", "van", "whale", "windmill"]

small_objs = random.sample(small_objs, len(big_objs))


all_shapes = small_objs + mid_objs + big_objs


first_list = big_objs if bigger_obj_first else small_objs
second_list = small_objs if bigger_obj_first else big_objs

all_sentences = list()

for first_obj in first_list:
    for second_obj in second_list:
        mid_objs = random.sample(mid_objs, n-2)
        current_comb = [first_obj] + mid_objs + [second_obj]
        all_sentences.append(" and ".join(current_comb))


shape_texts = tokenizer(all_shapes).to(device)
with torch.no_grad(), torch.cuda.amp.autocast():
    shape_features = model.encode_text(shape_texts)
shape_features /= shape_features.norm(dim=-1, keepdim=True)


topn_correct = 0
word_correct_in_topn = [0] * n
new_word_correct_in_topn = [0] * n

actual = list()
predicted = list()

for i in tqdm(range(0, len(all_sentences), batch_size), desc="Processing sentences"):
    batch_sentences = all_sentences[i:i + batch_size]
    text = tokenizer(batch_sentences).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        batch_sentence_features = model.encode_text(text)
        batch_sentence_features /= batch_sentence_features.norm(dim=-1, keepdim=True)

    cos_similarities = torch.mm(batch_sentence_features, shape_features.t()).cpu().numpy()

    for j, sentence in enumerate(batch_sentences):
        sentence_cos_similarities = cos_similarities[j]
        topn_indices = np.argsort(sentence_cos_similarities)[-n:]
        topn_words = [all_shapes[k] for k in topn_indices]
        words_in_sentence = sentence.split(' and ')

        actual.append(words_in_sentence)
        predicted.append(topn_words)

        most_similar_word_index = np.argmax(sentence_cos_similarities)
        most_similar_word = all_shapes[most_similar_word_index]

        for k in range(n):
            if most_similar_word == words_in_sentence[k]:
                new_word_correct_in_topn[k] += 1
                break

        if all(word in sentence for word in topn_words):
            topn_correct += 1

            for k in range(n):
                if most_similar_word == words_in_sentence[k]:
                    word_correct_in_topn[k] += 1
                    break


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


to_print = list()

topn_accuracy = topn_correct / len(all_sentences)
word_accuracies_in_topn = [correct / topn_correct if topn_correct else 0 for correct in word_correct_in_topn]
new_word_accuracies_in_topn = [correct / len(all_sentences) if len(all_sentences) else 0 for correct in new_word_correct_in_topn]

print(f'Top-{n} Accuracy: {topn_accuracy:.4f}')
to_print.append(topn_accuracy)
for i, accuracy in enumerate(word_accuracies_in_topn):
    print(f'Word {i+1} Accuracy in Correct Top-{n}: {accuracy:.4f}')
    to_print.append(accuracy)
for i, accuracy in enumerate(new_word_accuracies_in_topn):
    print(f'Word {i+1} Accuracy in Correct Top-{n}: {accuracy:.4f}')
    to_print.append(accuracy)

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
to_print.append(sum(precisions) / len(precisions))
print(f"Mean Recall: {sum(recalls) / len(recalls):.2f}")
to_print.append(sum(recalls) / len(recalls))
print(f"Mean F1 Score: {sum(f1_scores) / len(f1_scores):.2f}")
to_print.append(sum(f1_scores) / len(f1_scores))
print(f"Mean Average Precision (MAP): {map_score:.2f}")
to_print.append(map_score)
print(f"Mean NDCG: {mean_ndcg_score:.2f}")
to_print.append(mean_ndcg_score)

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
    to_print.append(sum(final_p_at_n[i]) / len(final_p_at_n[i]))
    to_print.append(sum(final_r_at_n[i]) / len(final_r_at_n[i]))

to_print = ['{:.4f}'.format(x) for x in to_print]
print(",".join(to_print))