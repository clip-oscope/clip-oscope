# import libraries
import argparse
import torch
from torch import nn
from PIL import Image
import numpy as np
import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split, Dataset
import open_clip
from torchvision import transforms
import matplotlib.pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description="IOC Experiments Setups")
parser.add_argument("--one_object_dataset_path", type=str, default="CoCo-OneObject", help="Path to one object dataset")
parser.add_argument("--many_objects_dataset_path", type=str, default="CoCo-FourObject-Middle-Big:4", help="Path and object count in format 'path:number'")
parser.add_argument("--model_name", type=str, default="ViT-B-32", help="Model name")
parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained model source")
parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing images")
args = parser.parse_args()


# Dataset paths for one_object and custom dataset. 
one_object_dataset_path = args.one_object_dataset_path
many_objects_dataset_path = {}
path, num_objects = args.many_objects_dataset_path.split(":")
many_objects_dataset_path[path] = int(num_objects)

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function for loading clip models
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


# Define Custom Dataset Class
class HeadDataset(Dataset):
    def __init__(self, model, preprocess, dataset_path, one_objects_classes, batch_size=128, device='cuda:0'):
        self.image_encoders_output = []
        self.images = []
        self.labels = []

        class_to_id = {class_: i for i, class_ in enumerate(one_objects_classes)}
        id_to_class = {class_to_id[class_name]: class_name for class_name in class_to_id.keys()}

        list_dir = os.listdir(f'./{dataset_path}')
        for i in tqdm(range(0, len(list_dir), batch_size), desc="Preparing Dataset"):
          
          images_name = list_dir[i:i + batch_size]
          classes = [image_name.replace('.png', '').split('_') for image_name in images_name]
          image_features = extract_image_features([f'./{dataset_path}/{image_name}' for image_name in images_name], model, preprocess).to(device)
          
          for image_feature in image_features:     
            self.image_encoders_output.append(image_feature)
          for class_ in classes:
              self.labels.append([class_to_id[c] for c in class_])
        self.lables = torch.tensor(self.labels, dtype = torch.long)


    def __len__(self):
        return len(self.lables)

    def __getitem__(self, index):
        return self.lables[index], self.image_encoders_output[index]

# Define Head class

class Head(nn.Module):
    def __init__(self, num_classes):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            nn.LazyLinear(num_classes)
        )

    def forward(self, clip_output):
        return self.head(clip_output)


# Train Functions
def report_results(epoch, total_losses, corrects, size):
  print(f"Epoch {epoch + 1}")
  for i, (total_loss, correct) in enumerate(zip(total_losses, corrects)):
    print(f"Object {i}, Loss: {total_loss / size}, Accuracy: {100 * correct / size} %")

def acc(outputs, label):
  label_pred = torch.argmax(outputs, dim=1)
  return (label_pred == label).sum().item()



def train_one_epoch(epoch, k, dataloader, heads, optimizers, criterion, show_logs=False):
  total_losses = [0.0 for i in range(k)]
  corrects = [0 for i in range(k)]
  for i, (labels, encodings) in enumerate(dataloader):
      labels, encodings  = labels.to(device), encodings.to(device)
      for j, (head, optimizer) in enumerate(zip(heads, optimizers)):
        head.train()
        label = labels[:, j] # b
        optimizer.zero_grad()
        outputs = head(encodings) # b x 18
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        total_losses[j] += loss.item()
        corrects[j] += acc(outputs, label)
  train_loss = np.array(total_losses) / len(dataloader.dataset)
  train_acc = np.array(corrects) / len(dataloader.dataset)
  if show_logs:
    print('test: ')
    report_results(epoch, total_losses, corrects, len(dataloader.dataset))
    print(50 * '-')
  return train_loss, train_acc



def test_one_epoch(epoch, k, dataloader, heads, criterion, show_logs=False):
  total_losses = [0.0 for i in range(k)]
  corrects = [0 for i in range(k)]
  with torch.no_grad():
    for i, (labels, encodings) in enumerate(dataloader):
      labels, encodings  = labels.to(device), encodings.to(device)
      for j, head in enumerate(heads):
        head.eval()
        label = labels[:, j]
        outputs = head(encodings)
        loss = criterion(outputs, label)
        total_losses[j] += loss.item()
        corrects[j] += acc(outputs, label)

    test_acc = np.array(corrects) / len(dataloader.dataset)
    test_loss = np.array(total_losses) / len(dataloader.dataset)
    if show_logs:
      print('test: ')
      report_results(epoch, total_losses, corrects, len(dataloader.dataset))
      print(50 * '-')
    return test_loss, test_acc



def test_and_train(n, model_name_full, dataset_path, epochs, train_dataloader, test_dataloader, heads, optimizers, criterion):

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_one_epoch(epoch, n, train_dataloader, heads, optimizers, criterion, show_logs=False)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss, test_acc = test_one_epoch(epoch, n, test_dataloader, heads, criterion, show_logs=False)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    train_accuracies = 100 * np.array(train_accuracies)
    test_accuracies = 100 * np.array(test_accuracies)

    print("IOC Results: \n")
    print(f'{model_name_full}: {dataset_path}')
    for i in range(n):
        print(f'Object {i + 1}: train:{round(train_accuracies[-1, i], 2)} test:{round(test_accuracies[-1, i], 2)}')


# One object names
one_objects_classes = [image_name.replace('.png', '') for image_name in os.listdir(f'./{one_object_dataset_path}')]


# Experiments
model, preprocess,  = load_model(model_name=args.model_name, pretrained=args.pretrained)

model_name_full = f'{args.model_name} - {args.pretrained}'
    
for dataset_path in many_objects_dataset_path.keys():

    n = many_objects_dataset_path[dataset_path]

    print("Preparing Dataset:")
    total_set = HeadDataset(model, preprocess, dataset_path, one_objects_classes)
    train_set, test_set = random_split(total_set, [0.8, 0.2])
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    print("Datasets Created Successfully")
    
    heads = [Head(len(one_objects_classes)).to(device) for i in range(n)]
    criterion = torch.nn.CrossEntropyLoss()
    optimizers = [torch.optim.Adam(head.parameters(), lr=1e-3) for head in heads]

    test_and_train(n, model_name_full, dataset_path, args.epochs, train_dataloader, test_dataloader, heads, optimizers, criterion)