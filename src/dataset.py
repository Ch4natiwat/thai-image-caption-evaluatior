from torch.utils.data import Dataset
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
        images = sample["images"]
        correct_image = sample["correct_image"]

        images = [
            transform.resize(image, (self.output_size, self.output_size))
            for image in images
        ]
        correct_image =  transform.resize(correct_image, (self.output_size, self.output_size))
        
        sample["images"] = images
        sample["correct_image"] = correct_image

        return sample
    
    
class Tokenize(object):
    
    def __init__(self, tokenizer):
        
        self.tokenizer = tokenizer
    
    def __call__(self, sample):
        captions = sample["captions"]
        correct_caption = sample["correct_caption"]
        
        captions = self.tokenizer(captions, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        correct_caption = self.tokenizer(correct_caption, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        
        correct_caption = {
            "input_ids": torch.squeeze(correct_caption["input_ids"]),
            "attention_mask": torch.squeeze(correct_caption["attention_mask"])
        }
        
        sample["captions"] = captions
        sample["correct_caption"] = correct_caption
        
        return sample
        


class ToTensor(object):

    def __call__(self, sample):
        images = sample["images"]
        correct_image = sample["correct_image"]
        labels = sample["labels"]
        
        images = [
            torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32))
            for image in images
        ]
        correct_image =  torch.from_numpy(correct_image.transpose((2, 0, 1)).astype(np.float32))
        labels = torch.argmax(torch.from_numpy(np.array(labels)))
        
        sample["images"] = images
        sample["correct_image"] = correct_image
        sample["labels"] = labels
        
        return sample
    

class ToDevice(object):
    
    def __init__(self, device):
        self.device = device
        
    def __call__(self, sample):
        
        images = sample["images"]
        correct_image = sample["correct_image"]
        labels = sample["labels"]
        captions = sample["captions"]
        correct_caption = sample["correct_caption"]
        
        images = [image.to(self.device) for image in images]
        correct_image = correct_image.to(self.device)
        labels = labels.to(self.device)
        correct_caption = {
            "input_ids": correct_caption["input_ids"].to(self.device),
            "attention_mask": correct_caption["attention_mask"].to(self.device)
        }
        captions = {
            "input_ids": captions["input_ids"].to(self.device),
            "attention_mask": captions["attention_mask"].to(self.device)
        }
        
        sample["images"] = images
        sample["correct_image"] = correct_image
        sample["labels"] = labels
        sample["captions"] = captions
        sample["correct_caption"] = correct_caption
        
        return sample


class SimilarityDetectorDataset(Dataset):

    def __init__(self, csv_file, image_dir, number_of_candidates=10, transform=None, device="cpu"):
        
        self.similar_caption_group = pd.read_csv(csv_file)
        self.number_of_similars = number_of_candidates - 1
        self.image_dir = image_dir
        self.transform = transform
        self.label_template = [1.] + [0.] * self.number_of_similars
        self.device = device


    def __len__(self):
        
        return len(self.similar_caption_group)


    def __getitem__(self, index, number_of_candidates=20):
        
        data_point = self.similar_caption_group.iloc[index]
        
        correct_caption = data_point.iloc[0]
        correct_image_path = os.path.join(self.image_dir, data_point.iloc[1])
        
        similar_captions = data_point.iloc[2 : number_of_candidates + 2]
        similar_image_paths = data_point.iloc[number_of_candidates + 2 : 2 * number_of_candidates + 2]
        similar_image_paths = [os.path.join(self.image_dir, similar_image_path) for similar_image_path in similar_image_paths]
        
        sample_list = list(zip(similar_captions, similar_image_paths))
        sample_list = rd.sample(sample_list, self.number_of_similars)
        similar_captions, similar_image_paths = zip(*sample_list)
        
        correct_image = io.imread(correct_image_path)
        similar_images = [io.imread(similar_image_path) for similar_image_path in similar_image_paths]
        
        captions = [correct_caption, *similar_captions]
        images = [correct_image, *similar_images]
        labels = copy.deepcopy(self.label_template)
        
        shuffle_list = list(zip(captions, images, labels))
        rd.shuffle(shuffle_list)
        captions, images, labels = zip(*shuffle_list)
        
        sample = {
            "correct_caption": correct_caption,
            "correct_image": correct_image,
            "captions": captions,
            "images": images,
            "labels": labels
        }

        if self.transform:
            sample = self.transform(sample)

        return sample