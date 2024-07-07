from ics_discriminator.discriminator import CaptionDiscriminator, ImageDiscriminator
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
import torch.nn as nn
import torch
import sys
import os

sys.path.append(Path(os.path.abspath(__file__)).parents[0])

from dataset import (
    CaptionDiscriminatorDataset, 
    ImageDiscriminatorDataset,
    get_transform
)


DATA_DIR = "path/to/data"
NUMBER_OF_CANDIDATES = 5
EPOCHS = 20
VALIDATION_STEP = 100

DATA_DIR = "data/annotations/sample_5000_train"


if __name__ == "__main__":

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    model = CaptionDiscriminator(number_of_candidates=NUMBER_OF_CANDIDATES, hidden_layer_size=256)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data_transforms = get_transform(device)
    train_dataset = CaptionDiscriminatorDataset(
        f"{DATA_DIR}/train.csv", f"{DATA_DIR}/images", 
        number_of_candidates=NUMBER_OF_CANDIDATES, 
        transform=data_transforms
    )
    validation_dataset = CaptionDiscriminatorDataset(
        f"{DATA_DIR}/val.csv", f"{DATA_DIR}/images", 
        number_of_candidates=NUMBER_OF_CANDIDATES, 
        transform=data_transforms
    )

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=8)

    try:
        for epoch in range(EPOCHS):
            
            model.train()
            
            for sample_index, samples in enumerate(train_dataloader):
                
                captions = samples["captions"]
                correct_image = samples["correct_image"]
                labels = samples["labels"]
                
                output = model(captions, correct_image)
                
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (sample_index + 1) % VALIDATION_STEP == 0 or sample_index == len(train_dataloader) - 1:
                    
                    losses = []
                    
                    model.eval()
                    
                    for samples in validation_dataloader:
                    
                        captions = samples["captions"]
                        correct_image = samples["correct_image"]
                        labels = samples["labels"]
                        
                        output = model(captions, correct_image)
                        loss = criterion(output, labels)
                        losses.append(loss.detach().cpu().item())
                        
                    print(sum(losses) / len(losses))
                    
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f"model_{epoch + 1}.pth"))
                    
    finally:
        torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f"model_{epoch + 1}.pth"))