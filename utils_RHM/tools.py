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
import cv2
from PIL import Image
from collections import OrderedDict
import numpy as np
from sklearn import metrics
from models.utils import custom_replace
from utils.metrics import custom_mean_avg_precision, subset_accuracy, hamming_loss, example_f1_score
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
warnings.filterwarnings('ignore')

category_info = {'Aeration':0, 'Discolouration_colour':1, 'Discolouration_Outfall':2,'Fish': 3,'Modified_Channel':4, 'Obsruction':5,
                 'Outfall':6, 'Outfall_Aeration':7, 'Outfall_Screen':8, 'Outfall_Spilling':9,
                 'Rubbish':10, 'Sensor':11, 'Wildlife_Algal':12, 'Wildlife_Birds':13,
                 'Wildlife_Others':14}


def process_directory(directory_path, output_file='_classes.csv'):
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

def process_dataset(img_dir,device):
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


def evaluate_model(data , model, device):
    num_labels = len(category_info)
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
    return all_predictions

def split_string(s, max_length):
    return '\n'.join([s[i:i+max_length] for i in range(0, len(s), max_length)])

def visualize_result(test_dataset,all_predictions,multiple = False , save = False):
    test_dataset = test_dataset.dataset
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    save_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if multiple == False:
        sample_idx = random.randrange(len(test_dataset))
        logits = all_predictions[sample_idx]
        probs = torch.sigmoid(logits)
        threshold = 0.5
        predicted_classes = np.where(probs >= threshold, 1, 0)
        predicted_class_indices = np.nonzero(predicted_classes)[0]
        print(f'Category classifified by the model:')
        predicted_labels = []
        for i in predicted_class_indices:
            category = list(category_info.keys())[list(category_info.values()).index(i)]
            probability = probs[i].item()
            print(f"{category} {probability*100:.2f}%")
            predicted_labels.append(f"{category} {probability*100:.2f}%")

        title = f"Predicted: {', '.join(predicted_labels)}"
        image = test_dataset[0]['image']
        unnormalized_image = image * std[:, None, None] + mean[:, None, None]
        unnormalized_image = unnormalized_image.permute(1, 2, 0)
        unnormalized_image = unnormalized_image.numpy()
        unnormalized_image = unnormalized_image.clip(0, 1)
        plt.imshow(unnormalized_image)
        plt.title(title)
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
            predicted_labels = []
            for idx in predicted_class_indices:
                category = list(category_info.keys())[list(category_info.values()).index(idx)]
                probability = probs[idx].item()
                predicted_labels.append(f"{category} {probability*100:.2f}%")
            title = f"Predicted: {split_string(', '.join(predicted_labels), 32)}"
            # predicted_labels = [list(category_info.keys())[list(category_info.values()).index(idx)] for idx in predicted_class_indices]
            # title = f"Predicted: {split_string(', '.join(predicted_labels), 32)}"
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


def preprocess_single_image(image_path):
    scale_size = 640
    crop_size = 576
    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        normTransform]) 
    image = Image.open(image_path).convert('RGB')
    image_transform = testTransform
    if image_transform:
        image = image_transform(image)
    return image

def yolo_predict(image_path, specific_classes, device):
    model = YOLO('./pretrained_model/object_detection.pt').to(device)
    results = model(image_path,verbose=False)
    yolo_boxes = []
    for result in results:
        for box, class_id in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = box[:4]
            class_id = class_id.item()
            if class_id in specific_classes:
                yolo_boxes.append((x1.item(), y1.item(), x2.item(), y2.item(), class_id))
    return yolo_boxes

import torchvision.models.segmentation as segmentation
def deeplab_predict(image_path, device):
    deeplab_model = segmentation.deeplabv3_resnet101(weights=segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    deeplab_model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1,1), stride=(1,1))
    deeplab_model.load_state_dict(torch.load('./pretrained_model/river_segmentation.pth'))
    deeplab_model.to(device)
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    deeplab_model.eval()
    with torch.no_grad():
        output = deeplab_model(input_tensor)['out']
    pred_mask = output[0].cpu().squeeze()
    return pred_mask > 0

def calculate_overlap(yolo_boxes, mask, class_coefficients):
    mask_np = mask.cpu().numpy()
    mask_area = np.sum(mask_np)
    overlap_ratios = {}

    for box in yolo_boxes:
        x1, y1, x2, y2, class_id = box
        box_mask = np.zeros_like(mask_np)
        box_mask[int(y1):int(y2), int(x1):int(x2)] = 1

        intersection = np.logical_and(box_mask, mask_np)
        overlap_area = np.sum(intersection)
        overlap_ratio = overlap_area / mask_area if mask_area > 0 else 0
        if int(class_id) not in overlap_ratios:
            overlap_ratios[int(class_id)] = 0
        overlap_ratios[int(class_id)] += overlap_ratio * class_coefficients.get(int(class_id), 1.0)
    return overlap_ratios

def calculate_score(image_path,device,predicted_labels):
    specific_classes = [0, 4, 5]
    class_names = {0: 'Aeration_Extent', 4: 'Proportion_Modification', 5: 'Obstruction'}
    merged_ratios = {'Aeration_Extent': 0, 'Proportion_Modification': 0, 'Obstruction': 0}
    yolo_boxes = yolo_predict(image_path, specific_classes, device)
    mask = deeplab_predict(image_path, device)
    class_coefficients = {0: 0.8, 4: 2, 5: 3}
    overlap_ratios = calculate_overlap(yolo_boxes, mask, class_coefficients)
    for class_id, ratio in overlap_ratios.items():
        class_name = class_names[int(class_id)]
        merged_ratios[class_name] += ratio
    for key, ratio in merged_ratios.items():
        for index, label in enumerate(predicted_labels):
            if key == 'Aeration_Extent' and label[0]== 'Aeration':
                print(f"{key}: {ratio}%")
            elif key == 'Proportion_Modification' and label[0]== 'Modified_Channel':
                print(f"{key}: {ratio}%")
            elif key == 'Obstruction' and label[0]== 'Obstruction':
                print(f"{key}: {ratio}%")
    

def label_helper(model, image_path , device):
    image = preprocess_single_image(image_path)
    image = image.unsqueeze(0).to(device)
    mask = torch.tensor([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1.])
    mask = mask.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred, _, _ = model(image,mask)
    pred = pred.squeeze(0)
    pred = pred.data.cpu()
    probs = torch.sigmoid(pred)
    threshold = 0.5
    predicted_classes = np.where(probs >= threshold, 1, 0)
    predicted_class_indices = np.nonzero(predicted_classes)[0]
    predicted_labels = []
    for idx in predicted_class_indices:
        category = list(category_info.keys())[list(category_info.values()).index(idx)]
        probability = probs[idx].item()
        predicted_labels.append([category, probability*100 ] )
    # print the predicted labels and its probability
    print(f'Category classifified by the model:')
    for i in predicted_labels:
        print(f"{i[0]} {i[1]:.2f}%")
    print('=====================================')
    detetction_model = YOLO('./pretrained_model/object_detection.pt')
    result = detetction_model.predict(image_path, verbose=False)
    image = cv2.imread(image_path)
    if result[0].boxes.xyxy.numel() == 0:
        print('WE NEED YOUR HELP! There is no pollution detected by our system')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()  
    else:
        print(f'Some detected pollutions suggested by our system ( objects that were not detected require your assistance):')
        category_info_palette = {
        'Aeration': (255, 0, 0), 'Discolouration_Colour': (0, 255, 0), 'Discolouration_Outfall': (0, 0, 255),
        'Fish': (255, 255, 0), 'Modified_Channel': (0, 255, 255), 'Obsruction': (255, 0, 255),
        'Outfall': (128, 0, 0), 'Outfall_Aeration': (0, 128, 0), 'Outfall_Screen': (0, 0, 128),
        'Outfall_Spilling': (128, 128, 0), 'Rubbish': (0, 128, 128), 'Sensor': (128, 0, 128),
        'Wildlife_Algal': (128, 128, 128), 'Wildlife_Birds': (64, 0, 0), 'Wildlife_Others': (0, 64, 0)
        }
        calculate_score(image_path,device,predicted_labels)
        for i, box in enumerate(result[0].boxes.xyxy):
            x_min, y_min, x_max, y_max = box[:4]
            class_id = result[0].boxes.cls[i]
            score = result[0].boxes.conf[i]
            class_name = list(category_info_palette.keys())[int(class_id)]
            for index, label in enumerate(predicted_labels):
                if class_name == label[0]: 
                    color = category_info_palette[class_name] 
                    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 4)
                    text = f"{class_name}: {score*100:.2f}%"
                    cv2.putText(image, text, (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 8)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()              




