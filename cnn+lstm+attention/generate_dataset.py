import scipy.io
import os, random
import numpy as np

train_set_size = 3200
test_set_size = 64

characters = os.listdir('character')
chars = [] # the gt char of each image
images = []
for file_name in characters:
    image = scipy.misc.imread('character/' + file_name)
    char = int(file_name[0])
    images.append(image)
    chars.append(char)

train_label_file = open('label/train_label.txt', 'w')

for step in range(train_set_size):
    new_image = np.ones((28, 182)) * 255
    digit_num = random.randint(2, 8)
    path = 'dataset/train/' + str(step) + '.png'
    train_label_file.write(path + ' ')
    for i in range(digit_num):
        c = random.randint(0, 39)
        new_image[:, i * 22:i * 22 + 28] = images[c]
        train_label_file.write(str(chars[c]))
    scipy.misc.toimage(new_image, high=255, low=0, cmin=0, cmax=255).save(path)
    train_label_file.write('\n')

train_label_file.close()

test_label_file = open('label/test_label.txt', 'w')

for step in range(test_set_size):
    new_image = np.ones((28, 182)) * 255
    digit_num = random.randint(2, 8)
    path = 'dataset/test/' + str(step) + '.png'
    test_label_file.write(path + ' ')
    for i in range(digit_num):
        c = random.randint(0, 39)
        new_image[:, i * 22:i * 22 + 28] = images[c]
        test_label_file.write(str(chars[c]))
    scipy.misc.toimage(new_image, high=255, low=0, cmin=0, cmax=255).save(path)
    test_label_file.write('\n')

test_label_file.close()