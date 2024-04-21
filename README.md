# Video Face Forgery Detection
Training the AltFreezing model to classify videos as tampered or original by detecting both spatial artifacts (e.g., generative artifacts, blending) and temporal artifacts (e.g., flickering, discontinuity) instead of focusing on just one type of artifact.


## Installation 

1. Install torch and torchvision: choose the version that suits your device in https://pytorch.org/

2. Install dependencies: 

```bash
pip install -r requirements.txt
```

## Training
### 1. Preprocess

Extract face boxes from videos in path ``{...}/videos`` and save to path 
``{...}/boxes``

```bash
usage: preprocess.py [-h] [--device DEVICE] [--frames FRAMES] input 

positional arguments:
  input            Path to video folder or video file

options:
  -h, --help       show this help message and exit
  --device DEVICE  Device index to select. Default: cuda:0
  --frames FRAMES  Number of frames to detect. Default: 1000
```

#### Example
```bash
python preprocess.py dataset/original_sequences/videos
```

### 2. Train
#### Example
```bash
python train.py 
```
Change training config in ``config.yaml`` such as batch size, accelerator, max epoch. For example:
```bash
dataloader:
  batch_size: 1
```

## Inference 

```bash
usage: predict.py [-h] [--device DEVICE] [--frames FRAMES] input output

positional arguments:
  input            Path to video folder or video file
  output           Path to output CSV file

options:
  -h, --help       show this help message and exit
  --device DEVICE  Device index to select. Default: cuda:0
  --frames FRAMES  Number of frames to detect. Default: 160
```

Notes: Video is stored in MP4 format: 

* `--device` (Optional): device to load the data into. (Default: `cpu`) 
* `--frames` (Optional): number of frames to detect in the whole video. (Default: `160`) 
* `input`: path to video folder or video file
* `output`: path to output CSV file

For example when the input is a single video

```bash
python predict.py example.mp4 statics.csv
```

For example when the input is a video folder

```bash
python predict.py example statics.csv
```

#### Explaination

1. Video after getting the same number of frames as the value of `--frames`
```
frames = read_video(filepath, max_frames)
```

2. Frames are then divided into batches with the size in the second parameter of the function.
```
batch_frames = partition(frames, 16)
```

3. Then the face image areas are detected by face detection model.
```
faces = get_valid_faces(flatten(faces), max_count=1, threshold=0.5)
```

4. Finally, the images are passed through the classification model 
