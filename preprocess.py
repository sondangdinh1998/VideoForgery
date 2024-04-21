import os
import glob
import argparse

import cv2
from tqdm import tqdm

from models.detection import load_detector, batch_detect, get_valid_faces
from models.alignment import CropAligner
from utils import read_video, partition, flatten, get_crop_box


def extract_face(filepath, max_frames, device):
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

    faces = [
        img
        for i in range(len(images))
        for img in aligner(images[i], boxes[i], landmarks[i])
    ]

    dirpath = filepath.replace("/videos", "/boxes")
    dirpath = dirpath.replace(".mp4", os.path.sep)
    os.makedirs(dirpath, exist_ok=True)

    for i, face in enumerate(faces):
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        path = os.path.join(dirpath, f"{i:0>4d}.jpg")
        cv2.imwrite(path, face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to video folder or video file")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device index to select"
    )
    parser.add_argument(
        "--frames", type=int, default=1000, help="Number of frames to detect"
    )
    args = parser.parse_args()

    if os.path.isfile(args.input):
        filepaths = [args.input]
    else:
        filepaths = glob.glob(f"{args.input}/**/*.mp4", recursive=True)

    device = args.device
    max_frames = args.frames

    detector = load_detector("checkpoints/detector.pth", device=device)
    aligner = CropAligner()

    for filepath in tqdm(filepaths):
        extract_face(filepath, max_frames, device)
