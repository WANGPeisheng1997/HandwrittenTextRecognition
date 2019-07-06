import scipy.io
import os, random
import numpy as np

train_set_size = 3200
val_set_size = 64
test_set_size = 1600

characters = os.listdir('character')
chars = [] # the gt char of each image
images = []
for file_name in characters:
    image = scipy.misc.imread('character/' + file_name)
    char = int(file_name[0])
    images.append(image)
    chars.append(char)


def data_generation(phase, size):
    label_file = open('label/%s_label.txt' % phase, 'w')
    for step in range(size):
        new_image = np.ones((28, 182)) * 255
        digit_num = random.randint(2, 8)
        path = 'dataset/%s/' % phase + str(step) + '.png'
        label_file.write(path + ' ')
        for i in range(digit_num):
            c = random.randint(0, 39)
            new_image[:, i * 22:i * 22 + 28] = images[c]
            label_file.write(str(chars[c]))
        scipy.misc.toimage(new_image, high=255, low=0, cmin=0, cmax=255).save(path)
        label_file.write('\n')

    label_file.close()


data_generation('train', train_set_size)
data_generation('val', val_set_size)
data_generation('test', test_set_size)
