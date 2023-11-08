import os
import glob
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from encodec import EncodecModel
from encodec.utils import convert_audio
import warnings
import torch.nn.functional as F
import open3d as o3d

from transformers import Wav2Vec2Processor
from FLAME_PyTorch.FLAME import FLAME
from FLAME_PyTorch.config import get_config
from utils import get_video_fps, get_video_length, read_video, save_video

config = get_config()
config.batch_size = 16


class MooflDataset(Dataset):
    def __init__(self,input_folder,in_memory=False,mode="train",read_audio=True, 
                                    file_name=False, return_exp=False):

        self.input_folder = input_folder
        self.model = EncodecModel.encodec_model_24khz()

        self.return_exp = return_exp
        self.in_memory = in_memory
        self.read_audio = read_audio
        self.mode = mode
        self.file_name= file_name

        self.audio_files = glob.glob(os.path.join(input_folder, "*.wav"))
        self.pose_files = [os.path.join(input_folder, f"{os.path.splitext(os.path.basename(f))[0]}_pose.npy") for f in self.audio_files]
        self.exp_files = [os.path.join(input_folder, f"{os.path.splitext(os.path.basename(f))[0]}_exp.npy") for f in self.audio_files]

        # find the longest and shortest pose file
        self.pose_lengths = [np.load(f).shape[0] for f in self.pose_files]
        self.max_pose_length = max(self.pose_lengths)
        self.min_pose_length = min(self.pose_lengths)

        ## find fps of the associated videos  video files have this extension _geometry_detail.mp4
        self.video_files = [os.path.join(input_folder, f"{os.path.splitext(os.path.basename(f))[0]}_video_gt_compressed.mp4") for f in self.audio_files]
        self.video_fps = [get_video_fps(f) for f in self.video_files]
        self.video_fps = np.array(self.video_fps)

        ## load the vertex decoder
        self.vertex_decoder = FLAME(config)
        for param in self.vertex_decoder.parameters():
            param.requires_grad = False

        ## check to see that the number of frames in pose data divided by the fps is equal to the length of the audio file
        self.pose_lengths = np.array(self.pose_lengths)
        self.pose_lengths = self.pose_lengths / self.video_fps

        # get video length
        self.video_lengths = [get_video_length(f) for f in self.video_files]
        self.video_lengths = np.array(self.video_lengths)
        self.video_lengths = self.video_lengths / self.video_fps

        assert np.allclose(self.pose_lengths, self.video_lengths, atol=1e-05)

        # only use datasamples coming from files with fps of 29 fps +- 1
        fps = 25
        self.pose_files = [f for i, f in enumerate(self.pose_files) if np.isclose(self.video_fps[i], fps, atol=1)]
        self.exp_files = [f for i, f in enumerate(self.exp_files) if np.isclose(self.video_fps[i], fps, atol=1)]
        self.audio_files = [f for i, f in enumerate(self.audio_files) if np.isclose(self.video_fps[i], fps, atol=1)]
        self.video_files = [f for i, f in enumerate(self.video_files) if np.isclose(self.video_fps[i], fps, atol=1)]
        self.pose_lengths = [f for i, f in enumerate(self.pose_lengths) if np.isclose(self.video_fps[i], fps, atol=1)]
        self.video_lengths = [f for i, f in enumerate(self.video_lengths) if np.isclose(self.video_fps[i], fps, atol=1)]
        self.video_fps = [f for i, f in enumerate(self.video_fps) if np.isclose(self.video_fps[i], fps, atol=1)]

        # load the wav2vec2 processor
        self.processor =  Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        processed_audio_files = []

        for i in range(len(self.pose_files)):
            # check if the pose data is already interpolated
            if os.path.exists(os.path.join(input_folder, f"{os.path.splitext(os.path.basename(self.audio_files[i]))[0]}_codes_wav2vec_wav.npy")) and os.path.exists(os.path.join(input_folder, f"{os.path.splitext(os.path.basename(self.audio_files[i]))[0]}.json")):
                processed_audio_files.append(os.path.join(input_folder, f"{os.path.splitext(os.path.basename(self.audio_files[i]))[0]}_codes_wav2vec_wav.npy"))
                continue
            else:
                audio_file = self.audio_files[i]
                audio_codes = self.audio_file_to_codes(audio_file)
                np.save(os.path.join(input_folder, f"{os.path.splitext(os.path.basename(audio_file))[0]}_codes_wav2vec_wav.npy"), audio_codes)
                processed_audio_files.append(os.path.join(input_folder, f"{os.path.splitext(os.path.basename(audio_file))[0]}_codes_wav2vec_wav.npy"))


        self.audio_files = processed_audio_files

        if self.in_memory:
            self.pose_files = [np.load(f) for f in self.pose_files]
            self.exp_files = [np.load(f) for f in self.exp_files]
            self.files = self.audio_files

            if self.read_audio:
                self.audio_files = [np.load(f) for f in self.audio_files]


        # load the template
        pcd = o3d.io.read_point_cloud("./dataset/model/FLAME_sample.ply")
        template = np.asarray(pcd.points).reshape((-1))
        self.template = torch.from_numpy(template).float()


    def __len__(self):
        return len(self.exp_files)
    

    def __getitem__(self,idx):
        # load npy files as torch tensors
        
        if not self.in_memory:
            exp = np.load(self.exp_files[idx])
            pose = np.load(self.pose_files[idx])
            audio_codes = np.load(self.audio_files[idx]) if self.read_audio else None
        else:
            exp = self.exp_files[idx]
            pose = self.pose_files[idx]
            audio_codes = self.audio_files[idx] if self.read_audio else None

    
        exp = torch.from_numpy(exp).float().unsqueeze(0)
        pose = torch.from_numpy(pose).float().unsqueeze(0)
        audio_codes = torch.from_numpy(audio_codes).float() if self.read_audio else None
        vertices = self.to_vertices(exp, pose).squeeze(0)

        if self.read_audio:
            return audio_codes, vertices, self.template, self.files[idx]

        if self.return_exp:
            return vertices, pose, exp
        
        return vertices, self.template, self.audio_files[idx]
        
    def audio_file_to_codes(self, audio_file):
        with warnings.catch_warnings():
            speech_array, sr = librosa.load(audio_file, sr=16000)
        input_values = np.squeeze(self.processor(speech_array,sampling_rate=16000).input_values)

        return input_values
    
    def to_vertices(self, exp, pose):
        shape_params = torch.zeros(exp.shape[0], exp.shape[1], 100)
        pose[:, : , :3] =   0
        vertex_out, _ = self.vertex_decoder(shape_params[:exp.shape[0]], exp, pose)
        vertex_out = vertex_out.reshape(exp.shape[0], -1, vertex_out.shape[1] * vertex_out.shape[2])
        return vertex_out



# Collate function for dataloader
def collate_fn_stage1(batch):
    vertices = [item[0] for item in batch]
    template = [item[1] for item in batch]
    files = [item[2] for item in batch]
    # Find the minimum length of the vertices
    min_len = min([item.shape[0] for item in vertices])

    # Truncate the vertices
    vertices = [item[:min_len, :] for item in vertices]
    # Convert to tensors
    vertices = torch.stack(vertices, dim=0)
    template = torch.stack(template, dim=0)
    return vertices, template, files

def collate_fn_stage2(batch):
    audio_codes = [item[0] for item in batch]
    vertices = [item[1] for item in batch]
    template = [item[2] for item in batch]
    files = [item[3] for item in batch]
    # Find the minimum length of the vertices
    min_len = min([item.shape[0] for item in vertices])
    # Truncate the vertices
    vertices = [item[:min_len, :] for item in vertices]

    # Find the minimum length of the audio codes
    min_len = min([item.shape[0] for item in audio_codes])
    # Truncate the audio codes
    audio_codes = [item[:min_len] for item in audio_codes]
    # Convert to tensors
    vertices = torch.stack(vertices, dim=0)
    template = torch.stack(template, dim=0)
    audio_codes = torch.stack(audio_codes, dim=0)
    return audio_codes, vertices, template, files


def get_dataloader_vox(config):
    dataset = MooflDataset("./VoxCeleb",
                            mode="train", read_audio=config.read_audio, file_name=config.visualize, in_memory=True)

    collate_fn = None
    if config.arch in ['vertices_encoder']:
        collate_fn = collate_fn_stage1 if config.batch_size > 1 else None
    elif config.arch in ['audio2vertices']:
        collate_fn = collate_fn_stage2 if config.batch_size > 1 else None

    dataloaders = create_dataloaders(dataset, config.batch_size, num_workers=10, shuffle=True, collate_fn=collate_fn)

    return dataloaders


def create_dataloaders(dataset, batch_size, num_workers=10, shuffle=True, collate_fn=None):
    dataset_len = len(dataset)
    train_size = int(dataset_len * 0.9)
    test_size = int((dataset_len-train_size)*0.5)
    valid_size = dataset_len - (train_size + test_size)

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    print(f"Train dataset size: ", train_size)
    print(f"Valid dataset size: ", valid_size)
    print(f"Test dataset size: ", test_size)
    dataset = {}

    dataset["train"]  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    dataset["valid"] = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True, collate_fn=collate_fn, drop_last=True)
    dataset["test"] = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn, drop_last=True)

    return dataset


def get_dataloader_ted(config):
    test_dataset = MooflDataset("./moofl/TED/emoca_25fps_processd/test_processed",
                            mode="test", read_audio=config.read_audio, in_memory=True)
    train_dataset = MooflDataset("./moofl/TED/emoca_25fps_processd/train_processed",
                            mode="train", read_audio=config.read_audio, in_memory=True)
    test_size = len(test_dataset)
    valid_size = int(0.2*test_size)

    test_dataset, valid_dataset =   torch.utils.data.random_split(test_dataset, [test_size-valid_size, valid_size])
    print(f"Train dataset size: ", len(train_dataset))
    print(f"Valid dataset size: ", valid_size)

    print(f"Test dataset size: ", test_size-valid_size)
    dataset = {}

        # create weighted sampler based on length of dataset.pose_lengths, weight should be such that longer sequences are sampled more often
    sampler = torch.utils.data.WeightedRandomSampler(train_dataset.pose_lengths, len(train_dataset.pose_lengths))
    dataset["train"]  = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler)
    dataset["valid"] = torch.utils.data.DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=True)
    dataset["test"] = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    return dataset


# load the vertex decoder
vertex_decoder = FLAME(config)
for param in vertex_decoder.parameters():
        param.requires_grad = False

def convert_to_vertices(exp, pose):
        shape_params = torch.zeros(exp.shape[0], exp.shape[1], 100)
        pose[:, : , :3] = 0
        vertex_out, _ = vertex_decoder(shape_params[:exp.shape[0]], exp.cpu(), pose.cpu())
        # vertexes get returned in shape (batch_size*seq_len, 5023, 3), where each entry in a batch has been concatenated in dim 0,  we need to reshape them to (batch_size, seq_len, 5023*3)
        vertex_out = vertex_out.reshape(exp.shape[0], -1, vertex_out.shape[1] * vertex_out.shape[2])
        return vertex_out

