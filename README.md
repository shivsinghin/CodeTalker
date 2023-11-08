## **CodeTalker**
A fork of codetalker repo.
This repo containes an audio to vertices/flame parameters model that generate realistic facial animation in a speed close to realtime.
<p align="center">
<img src="animation.gif" width="75%"/>
</p>

## **Environment**
- Linux
- Python 3.6+
- Pytorch 2.0

Other necessary packages:
```
pip install -r requirements.txt
```
- ffmpeg (rendering output videos)
- [Vertices2flame](http://github.com/simliai/vertices2flame) (Optional for flame blenshapes)

## **Dataset Preparation**
Download "FLAME_sample.ply" from [voca](https://github.com/TimoBolkart/voca/tree/master/template) and put it in `dataset/model`.
### VOCASET
Request the VOCASET data from [https://voca.is.tue.mpg.de/](https://voca.is.tue.mpg.de/). Place the downloaded files `data_verts.npy`, `raw_audio_fixed.pkl`, `templates.pkl` and `subj_seq_to_idx.pkl` in the folder `vocaset/`.  Read the vertices/audio data and convert them to .npy/.wav files stored in `vocaset/vertices_npy` and `vocaset/wav`:
```
cd vocaset
python process_voca_data.py
```

### VoxCeleb 
Request the Flame model from [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/). Place the download files `flame_static_embedding.pkl`, `flame_dynamic_embedding.npy` and `generic_model.pkl` in `dataset/model`.
Request the dataset from (TODO)

## **Demo**
Download the pretrained models and put the pretrained models under `vox` folder. Given the audio signal
```
sh scripts/download_weights.sh
```

- to animate a mesh in FLAME topology, run: 
	```
	sh scripts/demo.sh vox
	```

	This script will automatically generate the rendered videos in the `demo/output` folder. You can also put your own test audio file (.wav format) under the `demo/wav` folder and specify the arguments in `DEMO` section of `config/<dataset>/demo.yaml`.

## **Training**

The training operation shares a similar command:
```
sh scripts/train.sh <exp_name> config/<vox|vocaset>/<stage1|stage2>.yaml <vox|vocaset> <1|2|3>
```
Please replace `<exp_name>` with your own experiment name, `<vox|vocaset">` by the name of your target dataset, i.e., `vox` or `vocaset`. Change the `exp_dir` in `scripts/train.sh` if needed. We just take an example for default commands below.

### **Training for Discrete Motion Prior**

```
sh scripts/train.sh CodeTalker_s1 config/vox/stage1.yaml vox 1
```

### **Training for Speech-Driven Motion Synthesis**
Make sure the paths of pre-trained models are correct, i.e., `vqvae_pretrained_path` and `wav2vec2model_path` in `config/<vox|vocaset>/stage2.yaml`.
```
sh scripts/train.sh CodeTalker_s2 config/vox/stage2.yaml vox 2
```

## **Deployment**
### Launch server endpoint:
```
python server.py --config config/vox/demo.yaml
```


## **Acknowledgement**
We heavily borrow the code from
[FaceFormer](https://github.com/EvelynFan/FaceFormer),
[Learn2Listen](https://github.com/RenYurui/PIRender), and
[VOCA](https://github.com/TimoBolkart/voca). Thanks
for sharing their code and [huggingface-transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py) for their wav2vec2 implementation. We also gratefully acknowledge the ETHZ-CVL for providing the [B3D(AC)2](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html) dataset and MPI-IS for releasing the [VOCASET](https://voca.is.tue.mpg.de/) dataset. Any third-party packages are owned by their respective authors and must be used under their respective licenses.
