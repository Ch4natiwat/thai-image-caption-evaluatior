from dataset import SimilarityDetectorDataset, Rescale, ToTensor, Tokenize, ToDevice
from transformers import XLMRobertaTokenizer
from discriminator import Discriminator
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
import os


NUMBER_OF_CANDIDATES = 10
EPOCHS = 20
VALIDATION_STEP = 100


model = Discriminator(number_of_candidates=NUMBER_OF_CANDIDATES, hidden_layer_size=4096)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model)
  
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_transforms = transforms.Compose([
    Rescale(224),
    Tokenize(XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")),
    ToTensor(),
    ToDevice(device)
])
train_dataset = SimilarityDetectorDataset(
    "data/sample/train.csv", "data/sample/images", 
    number_of_candidates=NUMBER_OF_CANDIDATES, 
    transform=train_transforms
)
validation_dataset = SimilarityDetectorDataset(
    "data/sample/val.csv", "data/sample/images", 
    number_of_candidates=NUMBER_OF_CANDIDATES, 
    transform=train_transforms
)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=True, num_workers=8)

try:
    for epoch in tqdm(range(EPOCHS)):
        
        model.train()
        
        for sample_index, samples in enumerate(tqdm(train_dataloader, desc=f"Epoch: {epoch + 1} - Training")):
            
            correct_caption = samples["correct_caption"]
            correct_image = samples["correct_image"]
            captions = samples["captions"]
            images = samples["images"]
            labels = samples["labels"]
            
            caption_output, image_output = model(correct_caption, correct_image, captions, images)
            
            loss = criterion(caption_output, labels) + criterion(image_output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (sample_index + 1) % VALIDATION_STEP == 0 or sample_index == len(train_dataloader) - 1:
                
                losses = []
                
                model.eval()
                
                for samples in tqdm(validation_dataloader, desc="Validating"):
                
                    correct_caption = samples["correct_caption"]
                    correct_image = samples["correct_image"]
                    captions = samples["captions"]
                    images = samples["images"]
                    labels = samples["labels"]
                    
                    caption_output, image_output = model(correct_caption, correct_image, captions, images)
                    loss = criterion(caption_output, labels) + criterion(image_output, labels)
                    losses.append(loss.detach().cpu().item())
                    
                print(sum(losses) / len(losses))
                
        torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f"model_{epoch + 1}.pth"))
                
finally:
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f"model_{epoch + 1}.pth"))