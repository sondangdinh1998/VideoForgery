import os
import glob
import argparse

from tqdm import tqdm

import torch

from models.detection import load_detector, batch_detect, get_valid_faces
from models.alignment import CropAligner
from models.recognition import load_classifier
from utils import read_video, partition, flatten, get_crop_box


def detect_deepfake(filepath, max_frames, device):
    frames = read_video(filepath, max_frames)
    batch_frames = partition(frames, 16)

    faces = [batch_detect(detector, batch, device) for batch in batch_frames]
    faces = get_valid_faces(flatten(faces), max_count=1, threshold=0.5)

    items = []
    for i, face in enumerate(faces):
        if len(face) == 0:
            continue

        box, landmark, _ = face[0]
        image, box, landmark = get_crop_box(frames[i], box, landmark, 0.5)
        items.append((image, box, landmark))

    images, boxes, landmarks = map(lambda x: partition(x, 32), zip(*items))
    assert len(images) == len(boxes) == len(landmarks), "Length mismatch"

    preds = []
    for i in range(len(images)):
        if len(images[i]) != 32:
            continue

        batch = aligner(images[i], boxes[i], landmarks[i])
        batch = [torch.from_numpy(b).permute(2, 0, 1) for b in batch]

        batch = torch.stack(batch, dim=0).transpose(0, 1)
        batch = batch.unsqueeze(0).sub(mean).div(std).to(device)

        with torch.no_grad():
            score = classifier([batch]).sigmoid()

        preds.append(score.item())

    return sum(preds) / (len(preds) + 1e-9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to video folder or video file")
    parser.add_argument("output", help="Path to output CSV file")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device index to select"
    )
    parser.add_argument(
        "--frames", type=int, default=160, help="Number of frames to detect"
    )
    args = parser.parse_args()

    if os.path.isfile(args.input):
        filepaths = [args.input]
    else:
        filepaths = glob.glob(f"{args.input}/**/*.mp4", recursive=True)

    device = args.device
    max_frames = args.frames

    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
    mean = mean.view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
    std = std.view(1, 3, 1, 1, 1)

    detector = load_detector("checkpoints/detector.pth", device=device)
    aligner = CropAligner()
    classifier = load_classifier("checkpoints/classifier.pth", device=device)

    statistic = []
    for filepath in tqdm(filepaths):
        filename = os.path.basename(filepath)
        score = detect_deepfake(filepath, max_frames, device)
        statistic.append((filename, score))

    with open(args.output, "w") as f:
        f.write("FILENAME,SCORE\n")
        for filename, score in statistic:
            f.write(f"{filename},{score:.3f}\n")
