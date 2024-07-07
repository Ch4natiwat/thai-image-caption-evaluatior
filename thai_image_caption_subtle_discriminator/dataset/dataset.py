from transformers import XLMRobertaTokenizer
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform
import pandas as pd
import random as rd
import numpy as np
import torch
import copy
import os


class Rescale(object):

    def __init__(self, output_size: int):
        
        self.output_size = output_size

    def __call__(self, sample):
        
        if "images" in sample:
            images = sample["images"]
            images = [
                transform.resize(image, (self.output_size, self.output_size))
                for image in images
            ]
            sample["images"] = images
        
        if "correct_image" in sample:
            correct_image = sample["correct_image"]
            correct_image =  transform.resize(correct_image, (self.output_size, self.output_size))
            sample["correct_image"] = correct_image

        return sample
    
    
class Tokenize(object):
    
    def __init__(self, tokenizer):
        
        self.tokenizer = tokenizer
    
    def __call__(self, sample):
        
        if "captions" in sample:
            captions = sample["captions"]
            captions = self.tokenizer(captions, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            sample["captions"] = captions
        
        if "correct_caption" in sample:
            correct_caption = sample["correct_caption"]
            correct_caption = self.tokenizer(correct_caption, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            correct_caption = {
                "input_ids": torch.squeeze(correct_caption["input_ids"]),
                "attention_mask": torch.squeeze(correct_caption["attention_mask"])
            }
            sample["correct_caption"] = correct_caption
        
        return sample
        


class ToTensor(object):

    def __call__(self, sample):
        
        if "images" in sample:
            images = sample["images"]
            images = [
                torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32))
                if len(image.shape) == 3 else torch.from_numpy(np.stack((image, image, image), axis=0).astype(np.float32))
                for image in images
            ]
            sample["images"] = images
        
        if "correct_image" in sample:
            correct_image = sample["correct_image"]
            correct_image =  torch.from_numpy(correct_image.transpose((2, 0, 1)).astype(np.float32))
            sample["correct_image"] = correct_image
        
        labels = sample["labels"]
        labels = torch.argmax(torch.from_numpy(np.array(labels)))
        sample["labels"] = labels
        
        return sample
    

class ToDevice(object):
    
    def __init__(self, device):
        self.device = device
        
    def __call__(self, sample):
        
        if "images" in sample:
            images = sample["images"]
            images = [image.to(self.device) for image in images]
            sample["images"] = images
        
        if "correct_image" in sample:
            correct_image = sample["correct_image"]
            correct_image = correct_image.to(self.device)
            sample["correct_image"] = correct_image
        
        if "captions" in sample:
            captions = sample["captions"]
            captions = {
                "input_ids": captions["input_ids"].to(self.device),
                "attention_mask": captions["attention_mask"].to(self.device)
            }
            sample["captions"] = captions
        
        if "correct_caption" in sample:
            correct_caption = sample["correct_caption"]
            correct_caption = {
                "input_ids": correct_caption["input_ids"].to(self.device),
                "attention_mask": correct_caption["attention_mask"].to(self.device)
            }
            sample["correct_caption"] = correct_caption
        
        labels = sample["labels"]
        labels = labels.to(self.device)
        sample["labels"] = labels
        
        return sample
    
    
def get_transform(device):
    
    return transforms.Compose([
        Rescale(224),
        Tokenize(XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")),
        ToTensor(),
        ToDevice(device)
    ])


class CaptionDiscriminatorDataset(Dataset):

    def __init__(self, csv_file, image_dir, number_of_candidates=10, transform=None, device="cpu"):
        
        self.similar_caption_group = pd.read_csv(csv_file)
        self.number_of_similar_choices = (len(self.similar_caption_group.columns) - 2) // 2
        self.number_of_similars = number_of_candidates - 1
        self.image_dir = image_dir
        self.transform = transform
        self.label_template = [1.] + [0.] * self.number_of_similars
        self.device = device


    def __len__(self):
        
        return len(self.similar_caption_group)


    def __getitem__(self, index):
        
        data_point = self.similar_caption_group.iloc[index]
        
        correct_caption = data_point.iloc[0]
        correct_image_path = os.path.join(self.image_dir, data_point.iloc[1])
        
        similar_captions = data_point.iloc[2 : self.number_of_similar_choices + 2]      
        similar_captions = rd.sample(list(similar_captions), self.number_of_similars)
        
        correct_image = io.imread(correct_image_path)
        
        captions = [correct_caption, *similar_captions]
        labels = copy.deepcopy(self.label_template)
        
        shuffle_list = list(zip(captions, labels))
        rd.shuffle(shuffle_list)
        captions, labels = zip(*shuffle_list)
        
        sample = {
            "correct_image": correct_image,
            "captions": captions,
            "labels": labels
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
class ImageDiscriminatorDataset(Dataset):

    def __init__(self, csv_file, image_dir, number_of_candidates=10, transform=None, device="cpu"):
        
        self.similar_caption_group = pd.read_csv(csv_file)
        self.number_of_similar_choices = (len(self.similar_caption_group.columns) - 2) // 2
        self.number_of_similars = number_of_candidates - 1
        self.image_dir = image_dir
        self.transform = transform
        self.label_template = [1.] + [0.] * self.number_of_similars
        self.device = device


    def __len__(self):
        
        return len(self.similar_caption_group)


    def __getitem__(self, index):
        
        data_point = self.similar_caption_group.iloc[index]
        
        correct_caption = data_point.iloc[0]
        correct_image_path = os.path.join(self.image_dir, data_point.iloc[1])
        
        similar_image_paths = data_point.iloc[self.number_of_similar_choices + 2 : 2 * self.number_of_similar_choices + 2]
        similar_image_paths = [os.path.join(self.image_dir, similar_image_path) for similar_image_path in similar_image_paths]
        similar_image_paths = rd.sample(similar_image_paths, self.number_of_similars)
        
        correct_image = io.imread(correct_image_path)
        similar_images = [io.imread(similar_image_path) for similar_image_path in similar_image_paths]
        
        images = [correct_image, *similar_images]
        labels = copy.deepcopy(self.label_template)
        
        shuffle_list = list(zip(images, labels))
        rd.shuffle(shuffle_list)
        images, labels = zip(*shuffle_list)
        
        sample = {
            "correct_caption": correct_caption,
            "images": images,
            "labels": labels
        }

        if self.transform:
            sample = self.transform(sample)

        return sample