import os
import pandas as pd
import torch.nn.functional as F
from torchvision import transforms
from dataloaders.riverpollution_roboflow import rproboflowDataset 
from torch.utils.data import  DataLoader
from models import CTranModel
import torch
from tqdm import tqdm   
import random
from collections import OrderedDict
import numpy as np
from sklearn import metrics
from models.utils import custom_replace
from utils.metrics import custom_mean_avg_precision, subset_accuracy, hamming_loss, example_f1_score
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')

def process_directory(directory_path, category_info, output_file='_classes.csv'):
    image_ids = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):
            # image_id = os.path.splitext(filename)[0]
            image_id = filename
            image_ids.append(image_id)
    # Create a DataFrame
    df = pd.DataFrame(image_ids, columns=['filename'])
    for key in category_info.keys():
        df[key] = ''  # Initialize other columns with empty strings
    output_file = os.path.join(directory_path, output_file)
    print('Saving to', output_file)
    df.to_csv(output_file, index=False)

def process_dataset(img_dir,category_info,device):
    num_labels = len(category_info)
    test_known = 0 
    test_known_labels = int(test_known*0.01*num_labels)
    # ====================== Load Data ======================   
    scale_size = 640
    crop_size = 576
    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        normTransform]) 
    test_dataset = rproboflowDataset(
    img_dir = img_dir,
    flag='user_defined',
    image_transform=testTransform,
    known_labels=test_known_labels,
    testing=True
    )
    test_loader = DataLoader(test_dataset, batch_size = 10,shuffle=False)
    use_lmt = True
    pos_emb = False
    layers = 3
    heads = 4
    dropout = 0.1
    no_x_features = False
    model = CTranModel(num_labels,use_lmt,pos_emb,layers,heads,dropout,no_x_features)
    model = model.to(device)
    return test_loader , model


def load_saved_model(saved_model_name,model):
    checkpoint = torch.load(saved_model_name)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def evaluate_model(data , model, num_labels, device):
    model.eval()
    all_predictions = torch.zeros(len(data.dataset),num_labels).cpu()
    all_targets = torch.zeros(len(data.dataset),num_labels).cpu()
    all_masks = torch.zeros(len(data.dataset),num_labels).cpu()
    all_image_ids = []
    loss_labels = 'all'

    max_samples = 1000
    batch_idx = 0
    loss_total = 0
    unk_loss_total = 0
    print(data)
    for batch in tqdm(data,mininterval=0.5,leave=False,ncols=50):
        if batch_idx == max_samples:
            break
        labels = batch['labels'].float()
        images = batch['image'].float()
        mask = batch['mask'].float()
        unk_mask = custom_replace(mask,1,0,0)
        all_image_ids += batch['imageIDs']
        mask_in = mask.clone()
        with torch.no_grad():
            pred,int_pred,attns = model(images.to(device),mask_in.to(device))
        loss =  F.binary_cross_entropy_with_logits(pred.view(labels.size(0),-1),labels.to(device),reduction='none')
        if loss_labels == 'unk': 
            loss_out = (unk_mask.to(device)*loss).sum()
        else: 
            loss_out = loss.sum() 
        ## Updates ##
        loss_total += loss_out.item()
        unk_loss_total += loss_out.item()
        start_idx,end_idx=(batch_idx*data.batch_size),((batch_idx+1)*data.batch_size)
        if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            pred = pred.view(labels.size(0),-1)
        all_predictions[start_idx:end_idx] = pred.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()
        all_masks[start_idx:end_idx] = mask.data.cpu()
        batch_idx +=1
    return all_predictions,all_targets

def split_string(s, max_length):
    return '\n'.join([s[i:i+max_length] for i in range(0, len(s), max_length)])

def visualize_result(test_dataset,all_predictions,all_targets ,category_info,multiple = False , save = False):
    test_dataset = test_dataset.dataset
    max_samples = len(test_dataset)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    save_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if multiple == False:
        sample_idx = random.randint(0,len(test_dataset))
        logits = all_predictions[sample_idx]
        probs = torch.sigmoid(logits)
        threshold = 0.5
        predicted_classes = np.where(probs >= threshold, 1, 0)
        predicted_class_indices = np.nonzero(predicted_classes)[0]
        print(f'Category classifified by the model:')
        for i in predicted_class_indices:
            category = list(category_info.keys())[list(category_info.values()).index(i)]
            probability = probs[i].item()
            print(f"{category} {probability:.2f}%")
        
        image = test_dataset[0]['image']
        unnormalized_image = image * std[:, None, None] + mean[:, None, None]
        unnormalized_image = unnormalized_image.permute(1, 2, 0)
        unnormalized_image = unnormalized_image.numpy()
        unnormalized_image = unnormalized_image.clip(0, 1)
        plt.imshow(unnormalized_image)
        plt.show()
        if save == True:
            if not os.path.exists('result'):
                os.makedirs('result')
            plt.savefig(os.path.join('result', f'single_prediction_{save_time}.png'))
    elif multiple == True:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for i in range(10):
            image = test_dataset[i]['image']
            unnormalized_image = image * std[:, None, None] + mean[:, None, None]
            unnormalized_image = unnormalized_image.permute(1, 2, 0).numpy()
            unnormalized_image = unnormalized_image.clip(0, 1)
            logits = all_predictions[i]
            probs = torch.sigmoid(logits)
            threshold = 0.5
            predicted_classes = np.where(probs >= threshold, 1, 0)
            predicted_class_indices = np.nonzero(predicted_classes)[0]
            target_class_indices = np.where(all_targets[i] == 1)[0]
            predicted_labels = [list(category_info.keys())[list(category_info.values()).index(idx)] for idx in predicted_class_indices]
            title = f"Predicted: {split_string(', '.join(predicted_labels), 32)}"
            ax = axes[i // 5, i % 5]
            ax.imshow(unnormalized_image)
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        if save == True:
            if not os.path.exists('result'):
                os.makedirs('result')
            plt.savefig(os.path.join('result', f'multiple_prediction_{save_time}.png'))




