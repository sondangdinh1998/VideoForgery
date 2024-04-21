import cv2
import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank


def tformfwd(trans, uv):
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy


def tforminv(trans, uv):
    Tinv = inv(trans)
    xy = tformfwd(Tinv, uv)
    return xy


def findNonreflectiveSimilarity(uv, xy, options=None):
    options = {"K": 2}

    K = options["K"]
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))

    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U, rcond=-1)
        r = np.squeeze(r)
    else:
        raise Exception("cp2tform:twoUniquePointsReq")

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([[sc, -ss, 0], [ss, sc, 0], [tx, ty, 1]])

    T = inv(Tinv)

    T[:, 2] = np.array([0, 0, 1])

    return T, Tinv


def findSimilarity(uv, xy, options=None):
    options = {"K": 2}

    # Solve for trans1
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)

    # Solve for trans2

    # manually reflect the xy data across the Y-axis
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r, _ = findNonreflectiveSimilarity(uv, xyR, options)

    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    trans2 = np.dot(trans2r, TreflectY)

    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)

    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv


def get_similarity_transform(src_pts, dst_pts):
    trans, trans_inv = findSimilarity(src_pts, dst_pts)
    return trans, trans_inv


def get_similarity_transform_for_cv2(src_pts, dst_pts):
    trans, _ = get_similarity_transform(src_pts, dst_pts)
    cv2_trans = trans[:, 0:2].T
    return cv2_trans, trans


def estimiate_batch_transform(all_src_pts, tgt_pts):
    tgt_pts = np.repeat(tgt_pts[None, ...], len(all_src_pts), 0).reshape(
        -1, 2
    )
    src_pts = np.array(all_src_pts).reshape(-1, 2)
    tfm, trans = get_similarity_transform_for_cv2(src_pts, tgt_pts)
    return tfm, trans


class CropAligner:
    def __init__(self):
        self.std_points = np.array(
            [
                [75.10117125, 75.0568],
                [147.92155, 73.7958375],
                [111.62725, 119.875525],
                [79.35935, 152.863725],
                [146.3935375, 151.7016375],
            ]
        )

    def __call__(self, images, boxes, landmarks):
        boxes = np.array(boxes)
        landmarks = np.array(landmarks)

        left_top = boxes[:, :2].min(0)
        right_bottom = boxes[:, 2:].max(0)

        size = right_bottom - left_top
        w, h = size

        diff = boxes[:, :2] - left_top[None, ...]
        landmarks = landmarks + diff[:, None, :]

        tfm, _ = estimiate_batch_transform(landmarks, tgt_pts=self.std_points)

        images = [self.process(tfm, image, d, h, w) for image, d in zip(images, diff)]
        images = np.stack(images)

        return images

    def process(self, tfm, image, d, h, w):
        assert isinstance(image, np.ndarray)
        new_image = np.zeros((h, w, 3), dtype=np.uint8)
        x, y = d
        h, w = image.shape[:2]
        new_image[y : y + h, x : x + w] = image
        new_image = cv2.warpAffine(new_image, tfm, (224, 224))
        return new_image
