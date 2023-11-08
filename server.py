#!/usr/bin/env python
import json
import numpy as np
import torch 

import os
import cv2
import torch
import miniaudio
import numpy as np
import librosa
import pickle
import time
from transformers import Wav2Vec2Processor
from base.utilities import get_parser
from models import get_model
from vertices2flame import FlameInverter
from base.baseTrainer import load_state_dict

cfg = get_parser()

def load_model(cfg):
    model = get_model(cfg)
    inverter = FlameInverter(args=cfg)

    model = model.cuda()
    inverter = inverter.cuda()

    if os.path.isfile(cfg.model_path):
            print("=> loading checkpoint '{}'".format(cfg.model_path))
            checkpoint = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cpu())
            checkpoint_inv = torch.load(cfg.inverter_path, map_location=lambda storage, loc: storage.cpu())

            load_state_dict(model, checkpoint['state_dict'], strict=False)
            load_state_dict(inverter, checkpoint_inv['state_dict'], strict=False)

            print("=> loaded checkpoint '{}'".format(cfg.model_path))
    else:
            raise RuntimeError("=> no checkpoint flound at '{}'".format(cfg.model_path))

    model.eval()
    inverter.eval()

    return model, inverter

def get_template(cfg):
    subject = cfg.subject

    template_file = 'dataset/model/templates.pkl'
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    temp = templates[subject]

    template = temp.reshape((-1))
    template = np.reshape(template,(-1,template.shape[0]))
    template = torch.FloatTensor(template).to(device='cuda')

    return template




from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from uvicorn import run
import os
import random
import base64
import time
import asyncio
app = FastAPI()
port = 3000
client_count = 0
# Define a function to hanPdle incoming WebSocket messages
messages = []

model, inverter = load_model(cfg)
template = get_template(cfg)
processor = Wav2Vec2Processor.from_pretrained(cfg.wav2vec2model_path)

async def handle_message(data, websocket):
    # Convert the incoming data to a NumPy array
    y = np.array(miniaudio.mp3_read_f32(data).samples) 
    y = librosa.resample(y, orig_sr=44100, target_sr=16000)

    audio_feature = np.squeeze(processor(y,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device='cuda')

    with torch.no_grad():
        start = time.time()     
        c = 0
        length=10
        first = True
        t = time.time()
        for out in model.generate(audio_feature, template, length):
            
            with torch.no_grad():
                pose, exp = inverter(out)

            pose = pose.squeeze().cpu().detach().numpy().astype(np.float32)
            exp = exp.squeeze().cpu().detach().numpy().astype(np.float32)
            finalOut = np.concatenate((exp,pose),axis = 1)
            result = []
            if c ==0:
                result.extend([b"START"])
            result.extend([(finalOut.shape[0]).to_bytes(4,"little")])
            for f in finalOut:
                result.extend([(f.shape[0]).to_bytes(4,"little") , c.to_bytes(4,"little"),f.astype(np.float32).tobytes()])
                c += 1 

            x =  b''.join(result)
            await websocket.send_bytes(x)
        await websocket.send_bytes(b"DONE")
        print(c)
        await asyncio.gather(*messages)
        print("----------------------------------")
        print("Total time taken", time.time()-start)
        # print(input_data)

@app.websocket("/ws")   
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async def Listen():
        while True:
            # Receive
            try:
                data = await websocket.receive()
                payload = data.get("bytes")
                if payload is not None:  
                    await handle_message(payload, websocket)
            except WebSocketDisconnect:
                break
            # except Exception as e:
            #     print(e)
            #     break   
    tasks = [asyncio.ensure_future(Listen())]
    await asyncio.gather(*tasks)
    
if __name__ == "__main__":
    run("server:app", host="0.0.0.0", port=8081)



