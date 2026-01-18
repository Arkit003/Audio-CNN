import os
import sys
import modal
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset,DataLoader
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from model import AudioCNN

app = modal.App("audio-cnn")

image=(modal.Image.debian_slim()
       .pip_install_from_requirements("requirements.txt")
       .apt_install(['wget','unzip','ffmpeg','libsndfile1'])
       .run_commands([
           "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
           "cd /tmp && unzip esc50.zip",
           "mkdir -p /opt/esc50-data",
           "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
           "rm -rf /tmp/ESC-50-master /tmp/esc50.zip",
        ])
       .add_local_python_source("model")
        )

volume=modal.Volume.from_name('esc50-data',create_if_missing=True)
model_volume = modal.Volume.from_name('esc50-modal',create_if_missing=True)

class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_dir, split="train",transform=None):
        super().__init__()
        self.data_dir=Path(data_dir)
        self.metadata=pd.read_csv(metadata_dir)
        self.split=split
        self.transform=transform
        
        if split== "train":
            self.metadata=self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata=self.metadata[self.metadata['fold']==5]
            
        self.classes=sorted(self.metadata['category'].unique())
        
        self.class_to_idx={ cls:idx for idx,cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):# classs=ESC50Dataset() ,classs[0]
        row=self.metadata.iloc[idx]
        audio_path=self.data_dir / "audio"/row['filename']
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0]>1:# [channels,samples]->[2,44000]->[1,44000]
            waveform=torch.mean(waveform,dim=0,keepdim=True)#keeping the dimesions same
        
        if self.transform:
            spectogram=self.transform(waveform)
        else:
            spectogram=waveform
            
        return spectogram, row['label']
            
def mixup(x,y):
    '''
    Mixing up data to produce more real world sounds rather than pure dog or honk sound 
    putting like 80% of the dog sound and 20% honk sound 
    basically creating noise in the dataset 
    '''
    lam = np.random.beta(0.2,0.2)#blending percentage
    
    batch_size = x.size(0)#(batch_size, channels, length)
    
    index= torch.randperm(batch_size).to(x.device) #shuffing the batch, for cominations
    
    #(0.7*audio1)+(1-0.7*audio2)
    mixed_x= lam * x + (1-lam)*x[index,:]
    y_a, y_b = y,y[index]#true predictions
    
    return mixed_x, y_a, y_b, lam 

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''
    For calculating the loss of the mixup data
    counting lam % for a(dog bark) error and (1-lam%) for b(car honk ) error
    
    '''
    return lam *criterion(pred,y_a) + (1-lam)* criterion(pred, y_b)


@app.function(image=image,gpu="A10G",volumes={"/data":volume,"/model":model_volume},timeout=60*60*3)
def train():
    from datetime import datetime
    
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir=f"/model/tensorboard_logs/run_{timestamp}"
    writer=SummaryWriter(log_dir)
    
    
    esc50_dir=Path("/opt/esc50-data")
    
    train_transform= nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,#frequency resolution
            hop_length=512,#time resolution
            n_mels=128,#vertical “pixels” in the spectrogram
            f_min=0,
            f_max=11025,#Nyquist frequency (22050 / 2)
            ),#conveting raw audio file into melspectogram
        
        T.AmplitudeToDB(),#converting raw power to logscale(db(decibals))
        
        #these both below  works  similar to dropout,kind of reducing overfitting
        T.FrequencyMasking(freq_mask_param=30),#randomly mask upto 30 f
        T.TimeMasking(time_mask_param=80)#radomly masks time segments
        
    )
    val_transform= nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,#frequency resolution
            hop_length=512,#time resolution
            n_mels=128,#vertical “pixels” in the spectrogram
            f_min=0,
            f_max=11025,#Nyquist frequency (22050 / 2)
            ),#conveting raw audio file into melspectogram
        
        T.AmplitudeToDB(),#coinverting raw power to logscale(db(decibals)
    )
    
    train_dataset= ESC50Dataset(data_dir=esc50_dir,
                                metadata_dir=esc50_dir/"meta"/"esc50.csv",
                                split="train",
                                transform=train_transform
                                )
    val_dataset= ESC50Dataset(data_dir=esc50_dir,
                              metadata_dir=esc50_dir/"meta"/"esc50.csv",
                              split="test",
                              transform=val_transform
                              )
    
    print("Traning Samples: ",len(train_dataset))
    print("Validation Samples: ",len(val_dataset))

    train_dataloader= DataLoader(dataset=train_dataset,
                                 batch_size=32,
                                 shuffle=True,
                                 )
    test_dataloader= DataLoader(dataset=val_dataset,
                                 batch_size=32,
                                 shuffle=False,
                                 )
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)
    
    num_epochs=100
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)#basically rather than [1,0,0,0]->[0.9,0.025,0.025,0.025] giving some values to the others data as well
    optimizer= optim.AdamW(model.parameters(),lr=0.0005,weight_decay=0.01)
    
    #it will gonna constanly increase the lr one some cycle and decrase on the other cycle.
    schedular= OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1
        
    )
    
    best_accuracy=0.0
    #training
    print("Started Training")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss=0.0
        
        progress_bar=tqdm(train_dataloader,desc=f"Epoch {epoch+1}/{num_epochs}") #basically defing the iterable 
        
        for data,target in progress_bar:
            data,target= data.to(device),target.to(device)
            
            if np.random.random() > 0.7:
                data, target_a, target_b , lam =mixup(data, target)
                output= model(data)
                loss= mixup_criterion(criterion,output,target_a,target_b,lam)
            
            else:
                output=model(data)
                loss=criterion(output,target)
                
            optimizer.zero_grad()#setting all grads zero before backpropogation
            loss.backward()
            optimizer.step()
            schedular.step()
            
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss':f"{loss.item():.4f}"})
        
        avg_epoch_loss= epoch_loss/len(train_dataloader)
        writer.add_scalar("Loss/train",avg_epoch_loss,epoch)
        writer.add_scalar("Learning_Rate",optimizer.param_groups[0]['lr'],epoch)
        
        #validation after each epoch
        
        model.eval()
        
        correct=0
        total= 0
        val_loss=0
        
        with torch.no_grad(): #without updating and gradients during evaluation
            for data,target in test_dataloader:
                data,target = data.to(device), target.to(device)
                
                output=model(data)
                loss=criterion(output,target)
                val_loss+=loss.item()
                
                _,predicted = torch.max(output.data,1)# make all the values 0 rather than the predicted value
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        accuracy=100* correct/total
        avg_val_loss= val_loss/len(test_dataloader)
        
        writer.add_scalar("Loss/Validation",avg_epoch_loss,epoch)
        writer.add_scalar("Loss/Validation",avg_epoch_loss,epoch)
        
        print(
            f'Epoch: {epoch+1} Loss: {avg_epoch_loss:.4f}, Val Loss:{avg_val_loss:.4f}, Accuracy:{accuracy:.2f}%')
        
        if accuracy>best_accuracy:
            best_accuracy=accuracy
            torch.save({
                'model_state_dict': model.state_dict(),# all learned weights and biases (parameters )
                'accuracy':accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes,
            },
            "/model/best_model.pth"
            )
            print(f'New best model saved:{accuracy:.2f}%')
    
    writer.close()
    print(f'Training completed,Best accuracy: {best_accuracy:.2f}')
        
@app.local_entrypoint()
def main():
    train.remote()
