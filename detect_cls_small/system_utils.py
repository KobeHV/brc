import os

import albumentations as A
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from PIL import ImageDraw, ImageFont
from albumentations.pytorch import ToTensorV2
from skimage import io
import time

from resnet import resnet50

DIR_PATH = os.path.dirname(__file__) + "/"
threshold_background = 110
bounding_size = 40

# model_pth = DIR_PATH + 'pre-trained_model/dl_model/epoch-5-acc-0.99501.pth'
model_pth = DIR_PATH + 'pre-trained_model/dl_model/epoch-28-loss-0.0045224.pth'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def remove_background(image_array, threshold, bias):
    """ Sets background to be black.
    Args:
        image_array:
        threshold:
        bias:
    Returns:

    """
    rows, cols, channels = image_array.shape
    image_array_removed_background = image_array.copy()

    print("threshold:", threshold)

    mask = (image_array_removed_background[:, :, 0] > threshold) & (
            image_array_removed_background[:, :, 1] > threshold) & (
                   image_array_removed_background[:, :, 2] > threshold)
    image_array_removed_background[mask] = 0
    # for i in range(rows):
    #     for j in range(cols):
    #         if (image_array_removed_background[i][j][0] > 0 or image_array_removed_background[i][j][1] > 0 and
    #                 image_array_removed_background[i][j][2] > 0):
    #             print(image_array_removed_background[i][j][0], image_array_removed_background[i][j][1],
    #                   image_array_removed_background[i][j][2])

    mask2 = (image_array_removed_background[:, :, 2] < 100) | (image_array_removed_background[:, :, 1] < 80)
    # image_array_removed_background[:, :, 1] < 90) | (
    #         image_array_removed_background[:, :, 0] < 100)
    image_array_removed_background[mask2] = 0

    # for i in range(rows):
    #     for j in range(cols):
    #         if (image_array[i][j][0] > threshold and image_array[i][j][1] > threshold and image_array[i][j][
    #             2] > threshold
    #                 and abs(int(image_array[i][j][0]) - int(image_array[i][j][1])) <= bias
    #                 and abs(int(image_array[i][j][0]) - int(image_array[i][j][2])) <= bias
    #                 and abs(int(image_array[i][j][1]) - int(image_array[i][j][2])) <= bias):
    #             image_array_removed_background[i][j][0] = 0
    #             image_array_removed_background[i][j][1] = 0
    #             image_array_removed_background[i][j][2] = 0
    # elif (image_array[i][j][0] < 30 and image_array[i][j][1] < 45 and image_array[i][j][2] > 95
    #       or image_array[i][j][0] > 90 and image_array[i][j][1] < 65 and image_array[i][j][2] < 45
    #       or image_array[i][j][0] < 40 and image_array[i][j][1] > 95 and image_array[i][j][2] < 70):
    #     image_array_removed_background[i][j][0] = 0
    #     image_array_removed_background[i][j][1] = 0
    #     image_array_removed_background[i][j][2] = 0

    return image_array_removed_background


def grayscale_convert(image_array):
    """ Converts an image_array to a grayscale image_array(Only contains 'Height' and 'Width').
    Args:
        image_array:

    Returns:
    """
    image_array_grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return image_array_grayscale


def thresholding_convert(image_array, threshold):
    """ Uses Thresholding to convert an image_array.
    Args:
        image_array:
        threshold:

    Returns:

    """
    ret, image_array_thresholding = cv2.threshold(image_array, threshold, 255, cv2.THRESH_BINARY)
    return image_array_thresholding


def denoise_using_bilateral_filter(image_array):
    """ Uses 'Bilateral Filter' to smoothly denoise and preserve the edge.
    Args:
        image_array:

    Returns:z

    """
    image_array_denoised = cv2.bilateralFilter(image_array, d=9, sigmaColor=80, sigmaSpace=80)
    return image_array_denoised


def fill_using_flood_fill(image_array):
    """ Uses 'Flood Fill Algorithm' to fill the image_array.
    Args:
        image_array:

    Returns:

    """
    fill = image_array.copy()
    h, w = image_array.shape[: 2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(fill, mask, (90, 90), 255)
    fill_INV = cv2.bitwise_not(fill)
    fill_out = image_array | fill_INV
    # image_array = grayscale_convert(fill_out)
    # image_array = thresholding_convert(image_array, threshold=254)
    return fill_out


def bwareaopen_in_matlab(image_array, small_area_size):
    """ Removes all small white areas in black background.
    Args:
        image_array:
        small_area_size: size of small white areas.

    Returns:

    """
    # image_array = grayscale_convert(image_array)
    image_array_bwareaopened = image_array.copy()
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_array)
    for i in range(1, nlabels - 1):
        regions_size = stats[i, 4]
        if regions_size < small_area_size:
            x0 = stats[i, 0]
            y0 = stats[i, 1]
            x1 = stats[i, 0] + stats[i, 2]
            y1 = stats[i, 1] + stats[i, 3]
            for row in range(y0, y1):
                for col in range(x0, x1):
                    if labels[row, col] == i:
                        image_array_bwareaopened[row, col] = 0
    return image_array_bwareaopened


def replace_by_original_objects(marked_image_array, original_image_array):
    """ All white areas in bwareaopened image are replaced by original objects.
    Args:
        marked_image_array: That is, 'image_array_bwareaopened'.
        original_image_array:

    Returns:

    """
    rows, cols, channels = marked_image_array.shape
    image_array_replaced = original_image_array.copy()

    for i in range(rows):
        for j in range(cols):
            if (marked_image_array[i][j][0] == 0 and marked_image_array[i][j][1] == 0 and marked_image_array[i][j][
                2] == 0):
                image_array_replaced[i][j] = (255, 255, 255)

    return image_array_replaced


def detect_objects_contours(marked_image_array, original_image_array):
    """ Detects contours of all objects in the original image.
    Args:
        marked_image_array: That is, 'image_array_bwareaopened'.
        original_image_array:

    Returns:
        original_image_array: image with rectangles of detected objects.
    """
    marked_image_array = marked_image_array.copy()
    image_array_detected = original_image_array.copy()

    # marked_image_array = grayscale_convert(marked_image_array)
    marked_image_array = thresholding_convert(marked_image_array, threshold=127)

    contours, hierarchy = cv2.findContours(marked_image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max = 0
    maxA = 0
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > maxA:
            max = i
            maxA = w * h

    # cv2.drawContours(captured_images, contours, -1, (0, 255, 0), 5)

    contours_without_very_small = list(contours)

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w < bounding_size or h < bounding_size:
            contours_without_very_small.remove(contour)
        else:
            cv2.rectangle(image_array_detected, (x, y), (x + w, y + h), (255, 0, 0), 5)

    return image_array_detected, contours_without_very_small


def crop_and_label_objects(original_image_path, contours, cropped_image_save_root_path, cropped_image_name_prefix,
                           image_detected_path, detected_and_labeled_image_save_root_path,
                           detected_and_labeled_image_name_prefix):
    """ Crops and labels all objects in the original image.
    Args:
        original_image_path: a PIL.Image.Image
        contours:
        cropped_image_save_root_path:
        cropped_image_name_prefix:
        image_detected_path:
        detected_and_labeled_image_save_root_path:
        detected_and_labeled_image_name_prefix
    Returns:

    """
    original_image = Image.open(original_image_path)
    image_array_detected_and_labeled = cv2.imread(image_detected_path)

    print(f"Number of objects: {len(contours)}")

    list_total = []
    for i, contour in enumerate(contours):
        list_crop = []
        x, y, w, h = cv2.boundingRect(contour)
        print(f"Size of cropped object {i + 1} (H x W): {h} x {w}")

        cropped_size = 180
        cropped_x = x - ((cropped_size - w) // 2)
        cropped_y = y - ((cropped_size - h) // 2)

        out = original_image.crop((cropped_x, cropped_y, cropped_x + cropped_size, cropped_y + cropped_size))

        saved_path = f"{cropped_image_save_root_path}{cropped_image_name_prefix}_{str(i + 1)}_crop.png"

        out.save(saved_path, 'PNG')
        out_saved_abspath = os.path.abspath(saved_path)

        # string_category = predict_category_by_ml(
        #     cropped_image_path=f"{cropped_image_save_root_path}{cropped_image_name_prefix}_{str(i + 1)}_cropped.png")

        # predict
        start_score = time.time()
        string_category, score = predict_category_by_dl(
            cropped_image_path=f"{cropped_image_save_root_path}{cropped_image_name_prefix}_{str(i + 1)}_crop.png")
        end_score = time.time()
        print('score time: %s Seconds' % (end_score - start_score))

        image_array_detected_and_labeled = image_add_text(image_array=image_array_detected_and_labeled,
                                                          text_content=string_category + " " + score, x=x,
                                                          y=y - 30, text_color=(255, 0, 0), text_size=20)
        list_crop.append(out_saved_abspath)
        list_crop.append(string_category)
        list_crop.append(score)

        list_total.append(list_crop)

    cv2.imwrite(
        f"{detected_and_labeled_image_save_root_path}{detected_and_labeled_image_name_prefix}_detected_and_labeled.png",
        image_array_detected_and_labeled)

    print(f"Image {detected_and_labeled_image_name_prefix}_detected_and_labeled.png saved successfully!\n\n")
    return list_total


# ML Algorithm----------------------------------------------------------------------

# 计算sift特征
def calculate_sift_feature(img):
    # 将图像转化为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 设置图像sift特征关键点最大为200
    sift = cv2.xfeatures2d.SURF_create()
    # 计算图片的特征点和特征点描述
    keypoints, features = sift.detectAndCompute(gray, None)
    return features


# 计算特征向量
def calculate_feature_vector(features, centers):
    featVec = np.zeros((1, 80))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (80, 1)) - centers
        # axis=1按行求和，即求特征到每个中心点的距离
        sqSum = (diffMat ** 2).sum(axis=1)
        dist = sqSum ** 0.5
        # 升序排序
        sortedIndices = dist.argsort()
        # 取出最小的距离，即找到最近的中心点
        idx = sortedIndices[0]
        # 该中心点对应+1
        featVec[0][idx] += 1
    return featVec


def predict_category_by_ml(cropped_image_path):
    clf = joblib.load(DIR_PATH + "pre-trained_model/ml_model/svm_model.m")
    centers = np.load(DIR_PATH + "pre-trained_model/ml_model/svm_centers.npy")
    image = io.imread(cropped_image_path)
    features = calculate_sift_feature(image)
    featVec = calculate_feature_vector(features, centers)
    case = np.float32(featVec)
    res = clf.predict(case)
    class_flower = ["亳菊", "滁菊", "贡菊", "杭菊", "怀菊"]
    return class_flower[int(res[0])]


# DL Algorithm----------------------------------------------------------------------


model = resnet50(pretrained=False)
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, 5)

model.eval()
model.load_state_dict(torch.load(model_pth, map_location=device), strict=True)
model = model.to(device)


def inference(image, img_size=256):
    inference_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

    image = Image.open(image).convert('RGB')
    image = np.array(image)
    transformed = inference_transforms(image=image)
    image = transformed['image']
    image = image.reshape(1, 3, img_size, img_size)
    image = image.to(device)
    return image


def predict_category_by_dl(cropped_image_path):
    # image = Image.open(cropped_image_path).convert('RGB')
    image = inference(cropped_image_path)
    _, label = model(image)
    label = nn.functional.softmax(label)
    score = format(label.cpu().max().detach().numpy().item() * 100, '.2f')
    label = label.cpu().argmax(1).detach().numpy()[0]
    flower_categories = ["亳菊", "滁菊", "贡菊", "杭菊", "怀菊"]
    return flower_categories[label], score


# ----------------------------------------------------------------------------------

# TODO: fix font issue on Linux
def image_add_text(image_array, text_content, x, y, text_color=(0, 255, 0), text_size=10):
    image_array = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    image_draw = ImageDraw.Draw(image_array)
    font_style = ImageFont.truetype("font/simsun.ttc", text_size, encoding="utf-8")
    image_draw.text((x, y), text_content, text_color, font=font_style)
    return cv2.cvtColor(np.asarray(image_array), cv2.COLOR_RGB2BGR)


def detect_and_label_objects(original_image_path,
                             removed_background_image_save_root_path, removed_background_image_name_prefix,
                             cropped_image_save_root_path, cropped_image_name_prefix,
                             detected_image_save_root_path, detected_image_name_prefix,
                             detected_and_labeled_image_save_root_path, detected_and_labeled_image_name_prefix):
    """ Detects and labels all objects in the original image.
    Args:
        original_image_path:
        removed_background_image_save_root_path:
        removed_background_image_name_prefix:
        cropped_image_save_root_path:
        cropped_image_name_prefix:
        detected_image_save_root_path:
        detected_image_name_prefix:
        detected_and_labeled_image_save_root_path:
        detected_and_labeled_image_name_prefix:

    Returns:

    """
    # image_array shape: (1080, 1920, 3)
    original_image_array = cv2.imread(original_image_path)
    original_image = Image.open(original_image_path)

    image_array_removed_background = remove_background(original_image_array, threshold=threshold_background, bias=40)
    cv2.imwrite(
        f"{removed_background_image_save_root_path}{removed_background_image_name_prefix}_removed_background.png",
        image_array_removed_background)

    image_array_grayscale = grayscale_convert(image_array_removed_background)
    cv2.imwrite("./restore/grayscale.png", image_array_grayscale)

    image_array_thresholding = thresholding_convert(image_array_grayscale, threshold=20)
    cv2.imwrite("./restore/binary.png", image_array_thresholding)
    image_array_denoised = denoise_using_bilateral_filter(image_array_thresholding)
    image_array_filled = fill_using_flood_fill(image_array_denoised)
    cv2.imwrite("./restore/denoise_and_fill.png", image_array_filled)
    image_array_bwareaopened = bwareaopen_in_matlab(image_array_filled, small_area_size=200)
    cv2.imwrite("./restore/bwareaopen.png", image_array_bwareaopened)
    # image_array_replaced = replace_by_original_objects(marked_image_array=image_array_bwareaopened,
    #                                                    original_image_array=original_image_array)
    image_array_detected, contours = detect_objects_contours(marked_image_array=image_array_bwareaopened,
                                                             original_image_array=original_image_array)
    cv2.imwrite(f"{detected_image_save_root_path}{detected_image_name_prefix}_detected.png", image_array_detected)

    start_crop_and_label_objects = time.time()
    list_res = crop_and_label_objects(original_image_path=original_image_path, contours=contours,
                                      cropped_image_save_root_path=cropped_image_save_root_path,
                                      cropped_image_name_prefix=cropped_image_name_prefix,
                                      image_detected_path=f"{detected_image_save_root_path}{detected_image_name_prefix}_detected.png",
                                      detected_and_labeled_image_save_root_path=detected_and_labeled_image_save_root_path,
                                      detected_and_labeled_image_name_prefix=detected_and_labeled_image_name_prefix)
    end_crop_and_label_objects = time.time()
    print(
        'crop_and_label_objects time: %s Seconds' % (end_crop_and_label_objects - start_crop_and_label_objects))
    return list_res
