import modal
import base64
import soundfile as sf
import io
import numpy as np
import librosa
import requests

import torch
import torch.nn as nn
import torchaudio.transforms as T
from pydantic import BaseModel

from model import AudioCNN

app=modal.App("audio-cnn-inference")

image=(modal.Image.debian_slim()
       .pip_install_from_requirements("requirements.txt")
       .apt_install(['libsndfile1'])
       .add_local_python_source("model")
       
       )
model_volume = modal.Volume.from_name('esc50-modal')

class AudioProcessor:
    '''
    takes an audio and transforms(process) it.
    '''
    def __init__(self):
        self.transform=nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,#frequency resolution
            hop_length=512,#time resolution
            n_mels=128,#vertical “pixels” in the spectrogram
            f_min=0,
            f_max=11025,#Nyquist frequency (22050 / 2)
            ),#conveting raw audio file into melspectogram
        
        T.AmplitudeToDB(),#converting raw power to logscale(db(decibals)
    )
    def process_audio_chunk(self, audio_data):
        
        '''
        it converts raw numpy array data(audio_data) to Melspectogram tensor,
        and unsqueezes a channel dimension shape[channels,samples]
        
        params: audio_data - raw audio data in numpy array shape
        
        Returns: [1, mel_bins, time_frames] - melspectogram tensor
        '''
        waveform= torch.from_numpy(audio_data).float()#shape=[22050]
        waveform = waveform.unsqueeze(0)#adding channels , shape=[1,22050]
        
        spectogram = self.transform(waveform)#[mel_bins, time_frames]
        
        return spectogram.unsqueeze(0)#[1, mel_bins, time_frames]
  
class InferenceRequest(BaseModel):
    audio_data : str  
            
@app.cls(image=image,gpu='A10G',volumes={"/model":model_volume},scaledown_window=15)#scaledown_window makes the model stays for 15s on the server gpu after each request          
class AudioClassifier:
    @modal.enter()#loads the model one time when the class is called 
    def load_model(self):
        print("Loading model on enter")
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load('/model/best_model.pth', map_location=self.device)
        
        self.classes = checkpoint['classes']
        
        self.model=AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.audio_processor = AudioProcessor()
        print("Modal loaded on Enter")
        
    @modal.fastapi_endpoint(method="POST")
    def inference(self,request:InferenceRequest):
        #for production -> upload file to s3 -> infernce endpoint ->download from s3 bucket
        #we are doing ->
            #frontend ->send file directly ->inference endpoint 
        
        audio_bytes = base64.b64decode(request.audio_data)
        
        audio_data, sample_rate = sf.read(
            io.BytesIO(audio_bytes),dtype="float32"
        )
        
        if audio_data.ndim>1:#meaning not mono audio channel >1
            audio_data=np.mean(audio_data,axis=1)
            
        if sample_rate != 44100:#try with 22050 also if doesn't works
            audio_data=librosa.resample(y=audio_data,orig_sr=sample_rate,target_sr=44100)
            
        spectogram= self.audio_processor.process_audio_chunk(audio_data)
        spectogram = spectogram.to(self.device)
        
        
        with torch.no_grad():
            output, feature_maps=self.model(spectogram , return_feature_maps=True)
            
            output=torch.nan_to_num(output)#making nan values as 0 
            probablities = torch.softmax(output, dim =1)#dim =0  batch , dim=1 class (batch_size,num_class)
            
            top3_probs, top3_indicies = torch.topk(probablities[0],3)
            
            predictions= [{"class":self.classes[idx.item()], "confidence":prob.item()}
                          for prob,idx in zip(top3_probs,top3_indicies)]
            
            
            
            viz_data = {} #taking the each feature map and visualizing it in 2d heat maps
            for name, tensor in feature_maps.items():
                if tensor.dim()==4: 
                    #[batch_size =1(only one file passed during inference) ,channels , height , width]
                    
                    aggregated_tensor = torch.mean(tensor, dim=1) # [b,h,w]->[1,h,w]
                    #we cant display all the channels , so aggrgating with mean
                    
                    squeezed_tensor = aggregated_tensor.squeeze(0)#[h,w]
                    numpy_array = squeezed_tensor.cpu().numpy()
                    clean_array = np.nan_to_num(numpy_array)
                    viz_data[name]={
                        "shape": list(clean_array.shape),
                        "values": clean_array.tolist()
                    }
                    
            spectogram_np = spectogram.squeeze(0).squeeze(0).cpu().numpy()
            clean_spectogram = np.nan_to_num(spectogram_np)
            
            max_samples=8000
            waveform_sample_rate = 44100
            if len(audio_data) > max_samples:
                step = len(audio_data) // max_samples
                waveform_data = audio_data[::step]
            else:
                waveform_data = audio_data

        response = {
            "predictions":predictions,
            "visulization":viz_data,
            "input_spectogram":{
                "shape": list(clean_spectogram.shape),
                "values": clean_spectogram.tolist()
            },
            "waveform":{
                "values": waveform_data.tolist(),
                "sample_rate" : waveform_sample_rate,
                "duration" : len(audio_data) / waveform_sample_rate
            },
            
        }       
        
                        
        return response
    
@app.local_entrypoint()
def main():
    audio_data, sample_rate = sf.read("fl-mocking-birds-36124.mp3")
    buffer = io.BytesIO()#buffer lets audio as a file , without actually writing to a file(keeping it in ram)
    sf.write(buffer,audio_data,sample_rate,format="WAV")
    
    #bytes  → base64 encode → ascii bytes → decode → str
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")#converting bytes data to str
    payload = {"audio_data":audio_b64}
    
    server = AudioClassifier() #modal object
    url = server.inference.get_web_url()#creates an modal endpoint in modal container https://arkit-audio-classifier.modal.run

    response = requests.post(url,json=payload)
    response.raise_for_status()
    
    result = response.json()
    
    waveform_info = result.get("waveform", {})
    if waveform_info:
        values = waveform_info.get("values",{})
        print(f"First 10 values are {[round(v,4) for v in values[:10]]}...")
        print(f"Duration: {waveform_info.get('duration',0)}" )
    
    print("Top predictions:")
    for pred in result.get("predictions",[]):
        print(f"- {pred['class']}: {pred['confidence']:0.2%}")
    