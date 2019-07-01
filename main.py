import torch

from PIL import Image, ImageOps
import numpy as np


from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import LeNet
from predict import predict, preprocessing, grayscale
from detect import NMS, selective_search

def load_network(network, save_path):
    network.load_state_dict(torch.load(save_path))
    return network


# model = LeNet()
# model = load_network(model, "mnist_lenet.pt")
#
# for group_name in ['a','b','c','d']:
#     print("Group:" + group_name)
#     for i in range(10):
#         test(model, "test/%d%s.png" % (i, group_name), i)

def detect_and_predict(image, model):
    # preprocess
    image = ImageOps.invert(image)
    image = grayscale(image)
    image.convert("RGB").save("test/crop.png")

    # detect
    skimage = io.imread("test/crop.png")
    candidates = selective_search(skimage)
    candidates_boxs = []
    for x, y, w, h in candidates:
        candidates_boxs.append([x, y, x + w, y + h, w * h])
    candidates_boxs = np.array(candidates_boxs)
    keep = NMS(candidates_boxs, 0.7)

    # predict
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(skimage)

    predict_dict = {}
    for x1, y1, x2, y2, _ in candidates_boxs[keep]:
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image = ImageOps.invert(cropped_image)
        cropped_image = preprocessing(cropped_image)
        pred, confidence = predict(model, cropped_image)
        predict_dict[x1] = pred
        plt.text(x1, y1, "%d  %.3f" % (pred, confidence), color='r')


    sorted_dict = sorted(predict_dict.items(), key=lambda x: x[0])
    predict_string = ""
    for x1, char in sorted_dict:
        predict_string += str(char)

    plt.text(0, 0, predict_string, color='r')

    for x1, y1, x2, y2, area in candidates_boxs[keep]:
        print(x1, y1, x2, y2, area)
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()


model = LeNet()
model = load_network(model, "mnist_lenet.pt")
image = Image.open("test/string3.png").convert('L')
detect_and_predict(image, model)





