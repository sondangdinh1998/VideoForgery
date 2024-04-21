import random

import numpy as np


class TemporalDropout:
    def __init__(self, min_frames, max_frames, p):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.p = p

    def __call__(self, frames):
        if random.random() > self.p:
            return frames

        num_frames = random.randint(self.min_frames, self.max_frames)
        frame_idxs = random.sample(range(len(frames)), k=num_frames)

        new_frames = np.copy(frames)
        for idx in frame_idxs:
            new_frames[idx].fill(0)

        return new_frames


class TemporalRepeat:
    def __init__(self, min_frames, max_frames, p):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.p = p

    def __call__(self, frames):
        if random.random() > self.p:
            return frames

        num_frames = random.randint(self.min_frames, self.max_frames)
        frame_idxs = random.sample(range(len(frames)), k=num_frames)

        new_frames = []
        for idx, frame in enumerate(frames):
            if idx in frame_idxs:
                new_frames.extend([frame, frame])
            else:
                new_frames.append(frame)

        new_frames = np.array(new_frames[:len(frames)])

        return new_frames


# class ClipBlending:
#     def __init__(self, p):
#         self.p = p

#     def __call__(self, frames_1, frames_2):
#         shape = frames_1[0].shape
#         mask = np.random.rand(*shape)

#         assert len(frames_1) == len(frames_2), "Length mismatch"

#         new_frames = []
#         for idx in range(len(frames_1)):
#             frame = frames_1[idx] * mask + frames_2[idx] * (1 - mask)
#             new_frames.append(frame)

#         new_frames = np.array(new_frames).astype(np.uint8)

#         return new_frames


class Compose:
    def __init__(self, augments) -> None:
        self.augments = augments

    def __call__(self, frames):
        for augment in self.augments:
            frames = augment(frames)
        return frames
