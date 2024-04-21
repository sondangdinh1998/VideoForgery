import cv2
import numpy as np


def read_video(filepath, max_frames):
    capture = cv2.VideoCapture(filepath)
    ret = True

    frames = []
    while ret:
        ret, frame = capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) == max_frames:
                break

    capture.release()

    return frames


def flatten(list_items):
    return [item for sublist in list_items for item in sublist]


def partition(images, size):
    return [
        images[i: i + size] if i + size <= len(images) else images[i:]
        for i in range(0, len(images), size)
    ]


def upsize_box(shape, box, scale):
    height, width = shape

    box = np.rint(box).astype(int)
    new_box = box.reshape(2, 2)

    size = new_box[1] - new_box[0]
    diff = scale * size
    diff = diff[None, :] * np.array([-1, 1])[:, None]

    new_box = new_box + diff
    new_box[:, 0] = np.clip(new_box[:, 0], 0, width - 1)
    new_box[:, 1] = np.clip(new_box[:, 1], 0, height - 1)
    new_box = np.rint(new_box).astype(int)
    new_box = new_box.reshape(-1)

    return new_box


def get_crop_box(frame, box, landmark, scale):
    shape = frame.shape[:2]
    box = upsize_box(shape, box, scale)

    top_left = box[:2][None, :]
    landmark = landmark - top_left

    x1, y1, x2, y2 = box
    frame = frame[y1:y2, x1:x2]

    return frame, box, landmark
