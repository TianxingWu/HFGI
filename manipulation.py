import cv2
import dlib
import numpy as np
import scipy
import scipy.ndimage
import PIL.Image

from editings.ganspace import edit


def get_landmark(img, predictor, detector, return_largest=True):
    """get landmark with dlib
    :param img: np.ndarray
    :param predictor: dlib predictor
    :param detector: dlib detector
    :param return_largest: whether return the landmarks of the largest face or of all faces
    :return: np.array shape=(68, 2)
    """
    # detector = dlib.get_frontal_face_detector()

    # img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    if return_largest:
        # only return landmarks of the largest face
        det_areas = [(det.right()-det.left())*(det.bottom()-det.top()) for det in dets]
        # dets = [dets[det_areas.index(max(det_areas))]]
        det = dets[det_areas.index(max(det_areas))]
    else:
        raise ValueError("No support for multiple faces editing yet!")

    # for k, d in enumerate(dets):
    #     shape = predictor(img, d)
    shape = predictor(img, det)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def align_face(img, predictor, detector):
    """
    :param img: PIL Image
    :param predictor: dlib predictor
    :param detector: dlib detector
    :return: PIL Image
    """
    # predictor = dlib.shape_predictor("./experiment/test/shape_predictor_68_face_landmarks.dat")
    lm = get_landmark(np.array(img), predictor, detector, return_largest=True)

    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    quad_orig = np.copy(quad)
    # read image
    # img = PIL.Image.open(filepath)

    output_size = 256
    transform_size = 256
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(
            np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect"
        )
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(
            mask * 3.0 + 1.0, 0.0, 1.0
        )
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
        quad += pad[:2]

    # Transform.
    img = img.transform(
        (transform_size, transform_size),
        PIL.Image.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Return aligned image.
    # and corner points of the quad (relative to crop)
    return img, quad+0.5, quad_orig, crop, pad


def attach_face(face_img, orig_img, face_quad, orig_quad, crop, pad):
    face_img = np.array(face_img)
    orig_img = np.array(orig_img)

    # zero padding orig
    left = max(0, (pad[0] - crop[0]))
    top = max(0, (pad[1] - crop[1]))
    right = max(0, (pad[2] - (orig_img.shape[1] - crop[2])))
    bottom = max(0, (pad[3] - (orig_img.shape[0] - crop[3])))
    padded_img = cv2.copyMakeBorder(orig_img, top, bottom, left, right, cv2.BORDER_REFLECT)
    orig_quad = orig_quad + [left, top]

    # top left, bottom left, bottom right, top right
    src_pts = np.array([
        [0, 0],
        [0, face_img.shape[0]],
        [face_img.shape[1], face_img.shape[0]],
        [face_img.shape[1], 0],
    ], dtype=np.float32)

    dst_pts = np.float32(orig_quad)

    # transform and warp face image
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    padded_size = (padded_img.shape[1], padded_img.shape[0])
    warped_face = cv2.warpPerspective(face_img, M, dsize=padded_size, flags=cv2.INTER_NEAREST)

    # =========== Seamless Clone ===========
    mask = np.copy(warped_face)
    mask = cv2.fillConvexPoly(mask, np.int32(np.rint(face_quad)), (255, 255, 255), lineType=cv2.LINE_AA)
    center = tuple(np.int32(np.sum(orig_quad, axis=0) / 4))
    # mixed_clone = cv2.seamlessClone(warped_face, cropped_img, mask, center, cv2.MIXED_CLONE)
    normal_clone = cv2.seamlessClone(warped_face, padded_img, mask, center, cv2.NORMAL_CLONE)
    normal_clone = normal_clone[top:normal_clone.shape[0]-bottom, left:normal_clone.shape[1]-right]
    ret = PIL.Image.fromarray(normal_clone)
    return ret


def attach_face_cropped(face_img, cropped_img, quad):
    # convert to numpy
    face_img = np.array(face_img)
    cropped_img = np.array(cropped_img)

    # top left, bottom left, bottom right, top right
    src_pts = np.array([
        [0, 0],
        [0, face_img.shape[0]],
        [face_img.shape[1], face_img.shape[0]],
        [face_img.shape[1], 0],
    ], dtype=np.float32)

    dst_pts = np.float32(quad)

    # transform and warp face image
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped_img_size = (cropped_img.shape[1], cropped_img.shape[0])
    warped_face = cv2.warpPerspective(face_img, M, dsize=cropped_img_size, flags=cv2.INTER_NEAREST)


    # =========== Simple Mask
    # # get mask and attach
    # # cv2.fillConvexPoly(cropped_img, np.int32(np.rint(quad)), (0, 0, 0), lineType=cv2.LINE_AA)
    # # img_middle = cropped_img
    # # out = cv2.add(warped_face, img_middle)
    # gray = cv2.cvtColor(warped_face, cv2.COLOR_BGR2GRAY)
    # _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    # # mask_inv = cv2.bitwise_not(mask)

    # cropped_img = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)
    # out = cv2.add(warped_face, cropped_img)

    # =========== Seamless Clone
    mask = np.copy(warped_face)
    mask = cv2.fillConvexPoly(mask, np.int32(np.rint(quad)), (255, 255, 255), lineType=cv2.LINE_AA)
    center = tuple(np.int32(np.sum(quad, axis=0) / 4))
    # mixed_clone = cv2.seamlessClone(warped_face, cropped_img, mask, center, cv2.MIXED_CLONE)
    normal_clone = cv2.seamlessClone(warped_face, cropped_img, mask, center, cv2.NORMAL_CLONE)

    return normal_clone
    

if __name__=='__main__':
    # test()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./experiment/test/shape_predictor_68_face_landmarks.dat")

    img = PIL.Image.open("./experiment/test/155.jpg")
    face_img, face_quad, orig_quad, crop, pad = align_face(img, predictor, detector)

    edited_face_img = PIL.Image.open("./experiment/inference_results/smile/00001.jpg")
    cropped_img = PIL.Image.open("./experiment/test/cropped_img.png")
    # attach_face_cropped(edited_face_img, cropped_img, face_quad)
    edited_img = attach_face(edited_face_img, img, face_quad, orig_quad, crop, pad)
    edited_img.save("./experiment/inference_results/edited_img.jpg")

    # face_img.save("./experiment/test/155_aligned.jpg")