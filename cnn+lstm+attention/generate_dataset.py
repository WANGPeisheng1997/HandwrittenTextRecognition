import scipy.io
import os, random
import numpy as np
from PIL import Image

train_set_size = 100
val_set_size = 64
test_set_size = 1600

characters = os.listdir('character')
chars = [] # the gt char of each image
images = []
for file_name in characters:
    image = Image.open('character/' + file_name)
    char = int(file_name[0])
    images.append(image)
    chars.append(char)


def data_generation(phase, size):
    label_file = open('label/%s_label.txt' % phase, 'w')
    for step in range(size):
        new_image = np.ones((32, 208)) * 255
        digit_num = random.randint(2, 6)
        path = 'dataset/%s/' % phase + str(step) + '.png'
        label_file.write(path + ' ')

        estimate_width = 27 * digit_num
        init_position = int(104 - estimate_width / 2 + random.randint(-10, 10))
        current_position = init_position

        for i in range(digit_num):
            c = random.randint(0, 39)
            choosen_image = images[c].resize((36, 36), Image.ANTIALIAS)
            angle = random.randint(-10, 10)
            choosen_image = choosen_image.rotate(angle, resample=Image.BILINEAR, fillcolor='white')
            rand_x1 = random.randint(4, 6)
            rand_y1 = random.randint(0, 2)
            rand_x2 = random.randint(31, 33)
            rand_y2 = rand_y1 + 32
            box = (rand_x1, rand_y1, rand_x2, rand_y2)
            choosen_image = choosen_image.crop(box)
            image_width = choosen_image.size[0]

            choosen_image.save('temp.png')
            image = scipy.misc.imread('temp.png')

            new_image[:, current_position:current_position+image_width] = image
            current_position += image_width
            label_file.write(str(chars[c]))

        scipy.misc.toimage(new_image, high=255, low=0, cmin=0, cmax=255).save(path)
        label_file.write('\n')

    label_file.close()


data_generation('train', train_set_size)
# data_generation('val', val_set_size)
# data_generation('test', test_set_size)
