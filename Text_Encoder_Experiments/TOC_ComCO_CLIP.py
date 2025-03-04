import os
import random
import torch
import pandas as pd
import torchvision
import tarfile
import open_clip
from PIL import Image
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
matplotlib.rcParams['figure.facecolor'] = '#ffffff'


model_type = "ViT-B-32"
pretrained_from = "laion2b_s34b_b79k"
embed_size = 512
variation_domain_index = 0 # 1 -> firt object | 3 -> second object | 5 -> third object | 7 -> forth object
total_number_of_sentences = 180000


random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)


clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(model_type, pretrained=pretrained_from)
clip_model.eval()
tokenizer = open_clip.get_tokenizer(model_type)


all_shapes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter",
               "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "dining table", "cell phone", 
               "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "kite",
               "baseball bat", "baseball glove", "tennis racket", "wine glass", "hot dog", "potted plant", "teddy bear", "hair drier", "hair brush", 
               "skateboard", "surfboard", "bottle", "plate", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
               "pizza", "donut", "cake", "chair", "couch", "bed", "mirror", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "microwave",
               "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "toothbrush"]
all_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'cyan', 'magenta', 'maroon', 'olive', 'navy', 'teal', 'lime', 'indigo', 'violet', 'gold', 'silver', 'coral', 'salmon', 'turquoise', 'beige', 'lavender', 'crimson', 'aqua', 'chartreuse']


all_possible_templates = set()

if variation_domain_index % 2 == 0:
    for color in all_colors:
        counter = 0
        while counter < 3000:
            four_random_shapes = random.sample(all_shapes, 4)
            index_of_color = all_colors.index(color)
            three_random_colors = random.sample(all_colors[:index_of_color] + all_colors[index_of_color+1:], 3)
            constructor = "color_shape_color_shape_color_shape_color_shape"
            constructor_split = constructor.split("_")
            constructor_split[variation_domain_index] = color
            for i, cons in enumerate(constructor_split):
                if cons == "color":
                    constructor_split[i] = three_random_colors[0]
                    del three_random_colors[0]
                elif cons == "shape":
                    constructor_split[i] = four_random_shapes[0]
                    del four_random_shapes[0]
            constructor = "_".join(constructor_split)
            if not constructor in all_possible_templates:
                all_possible_templates.add(constructor)
                counter += 1
else:
    for shape in all_shapes:
        counter = 0
        while counter < 3000:
            four_random_colors = random.sample(all_colors, 4)
            index_of_shape = all_shapes.index(shape)
            three_random_shapes = random.sample(all_shapes[:index_of_shape] + all_shapes[index_of_shape+1:], 3)
            constructor = "color_shape_color_shape_color_shape_color_shape"
            constructor_split = constructor.split("_")
            constructor_split[variation_domain_index] = shape
            for i, cons in enumerate(constructor_split):
                if cons == "color":
                    constructor_split[i] = four_random_colors[0]
                    del four_random_colors[0]
                elif cons == "shape":
                    constructor_split[i] = three_random_shapes[0]
                    del three_random_shapes[0]
            constructor = "_".join(constructor_split)
            if not constructor in all_possible_templates:
                all_possible_templates.add(constructor)
                counter += 1

all_possible_templates = list(all_possible_templates)


unique_variation_domain = set()

for template in all_possible_templates:
    template_split = template.split("_")
    variable = template_split[variation_domain_index]
    unique_variation_domain.add(variable)


random.shuffle(all_possible_templates)

sentences_for_each_label = total_number_of_sentences // len(unique_variation_domain)

final_dictionary = {}
for variable in list(unique_variation_domain):
    final_dictionary[variable] = list()

def check_end_of_sampling(sample_dict, number_of_sentences):
    for key, value in sample_dict.items():
        if len(value) != number_of_sentences:
            return False
    return True

for template in all_possible_templates:
    template_split = template.split("_")
    variable = template_split[variation_domain_index]
    if len(final_dictionary[variable]) != sentences_for_each_label:
        sentence = f"a {template_split[1]} and a {template_split[3]} and a {template_split[5]} and a {template_split[7]}"
        final_dictionary[variable].append(sentence)
    if check_end_of_sampling(final_dictionary, sentences_for_each_label):
        break


variable2label = dict()

for i, var in enumerate(unique_variation_domain):
    variable2label[var] = i


all_pairs = list()

for key, value in final_dictionary.items():
    for sen in value:
        all_pairs.append((key, sen))


class CustomDataset(Dataset):

    def __init__(self, sentence_pairs, tokenizer, mapper):
        self.sentence_pairs = sentence_pairs
        self.tokenizer = tokenizer
        self.mapper = mapper

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        current_pair = self.sentence_pairs[idx]
        tokenized_sentence = self.tokenizer([current_pair[1]])
        return tokenized_sentence.squeeze(), self.mapper[current_pair[0]]
    

main_dataset = CustomDataset(all_pairs, tokenizer, variable2label)


val_size = int(len(main_dataset) * 0.1)
test_size = int(len(main_dataset) * 0.15)
train_size = len(main_dataset) - val_size - test_size

train_ds, val_ds, test_ds = random_split(main_dataset, [train_size, val_size, test_size])
len(train_ds), len(val_ds), len(test_ds)


batch_size=32
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size*2)
test_dl = DataLoader(test_ds, batch_size*2)


all_unique_labels = len(unique_variation_domain)


@torch.no_grad()
def evaluate(model, val_loader):
    global clip_model
    model.eval()
    outputs = list()
    for batch in val_loader:
        clip_model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = clip_model.encode_text(batch[0])
            text_features /= text_features.norm(dim=-1, keepdim=True)
        out = model.validation_step((text_features.type(torch.float), batch[1]))
        outputs.append(out)
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    global clip_model
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            clip_model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = clip_model.encode_text(batch[0])
                text_features /= text_features.norm(dim=-1, keepdim=True)
            loss = model.training_step((text_features.type(torch.float), batch[1]))
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


class ClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class MainCustomModel(ClassificationBase):
    def __init__(self, number_of_classes):
        super().__init__()
        global embed_size
        self.network = nn.Sequential(
            nn.Linear(embed_size, number_of_classes)
        )

    def forward(self, xb):
        return self.network(xb)
    

model = MainCustomModel(all_unique_labels)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
    

device = get_default_device()


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)


to_device(clip_model, device)


num_epochs = 10
opt_func = torch.optim.Adam
lr = 0.001


history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
plot_accuracies(history)


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
plot_losses(history)


test_dl = DeviceDataLoader(test_dl, device)
result = evaluate(model, test_dl)
print(result)