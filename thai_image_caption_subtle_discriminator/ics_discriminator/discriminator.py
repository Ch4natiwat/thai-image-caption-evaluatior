from transformers import ViTFeatureExtractor, ViTModel
from transformers import XLMRobertaModel
from torch import nn
import torch


class CaptionExtractor(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        
    def forward(self, tokenized_text):
        
        outputs = self.model(**tokenized_text)   
        features = outputs.last_hidden_state
        
        return features
    
    
class ImageExtractor(nn.Module):
    
    def __init__(self):
        
        super().__init__()        
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
    def forward(self, image):
        
        device = image.device
        inputs = self.feature_extractor(images=image, do_rescale=False, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state
            
        return features
    
    
class CaptionDiscriminator(nn.Module):
    
    def __init__(self, image_extractor, caption_extractor, number_of_candidates, hidden_layer_size):
        
        self.number_of_candidates = number_of_candidates
        
        super().__init__()
        
        self.image_extractor = image_extractor
        self.caption_extractor = caption_extractor
        
        self.flatten_caption = nn.Flatten()
        self.fc1_caption = nn.Linear(393216 * number_of_candidates + 151296, hidden_layer_size)
        self.relu1_caption = nn.ReLU()
        self.fc2_caption = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.relu2_caption = nn.ReLU()
        self.classifier_caption = nn.Linear(hidden_layer_size, number_of_candidates)
        self.softmax_caption = nn.Softmax(dim=1)
        
        
    def forward(self, captions, correct_image):
        
        caption_features = []
        for candidate_index in range(self.number_of_candidates):
            tokenized_captions = {
                "input_ids": torch.squeeze(captions["input_ids"][:, candidate_index, :], 1),
                "attention_mask": torch.squeeze(captions["attention_mask"][:, candidate_index, :], 1)
            }
            caption_features.append(self.caption_extractor(tokenized_captions))
        
        correct_image_feature = self.image_extractor(correct_image)
        caption_discriminator_features = [correct_image_feature] + caption_features
        caption_discriminator_features = [self.flatten_caption(feature) for feature in caption_discriminator_features]
        caption_discriminator_features = torch.cat(caption_discriminator_features, dim=1)
        
        caption_output = self.relu1_caption(self.fc1_caption(caption_discriminator_features))
        caption_output = self.relu2_caption(self.fc2_caption(caption_output))
        caption_output = self.softmax_caption(self.classifier_caption(caption_output))
        
        return caption_output
    
    
class ImageDiscriminator(nn.Module):
    
    def __init__(self, image_extractor, caption_extractor, number_of_candidates, hidden_layer_size):
        
        super().__init__()
        
        self.image_extractor = image_extractor
        self.caption_extractor = caption_extractor
        
        self.flatten_image = nn.Flatten()
        self.fc1_image = nn.Linear(151296 * number_of_candidates + 393216, hidden_layer_size)
        self.relu1_image = nn.ReLU()
        self.fc2_image = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.relu2_image = nn.ReLU()
        self.classifier_image = nn.Linear(hidden_layer_size, number_of_candidates)
        self.softmax_image = nn.Softmax(dim=1)
        
        
    def forward(self, images, correct_caption):
        
        image_features = [self.image_extractor(image) for image in images]
        correct_caption_feature = self.caption_extractor(correct_caption)
        image_discriminator_features = [correct_caption_feature] + image_features
        image_discriminator_features = [self.flatten_image(feature) for feature in image_discriminator_features]
        image_discriminator_features = torch.cat(image_discriminator_features, dim=1)
        
        image_output = self.relu1_image(self.fc1_image(image_discriminator_features))
        image_output = self.relu2_image(self.fc2_image(image_output))
        image_output = self.softmax_image(self.classifier_image(image_output))
        
        return image_output