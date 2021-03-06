import PIL.Image
import os
from resizeimage import resizeimage
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
original_image_annotation = os.path.join(data_dir, 'original_image_annotation')

def resize_file(in_file, out_file, size):
    image = resizeimage.resize_thumbnail(PIL.Image.open(in_file), size)
    image.save(out_file)
    image.close()


def convertRGB(file, output):
    path_to_image = os.path.join(data_dir, file)
    rgba_image = PIL.Image.open(path_to_image)
    rgb_image = rgba_image.convert('RGB')
    rgb_image.save(output)

def sumImage(im1, im2, resize=False, output_name='silouette.png'):
    im1arr = np.asarray(im1)
    im2arr = np.asarray(im2)
    addition = im1arr + im2arr
    resultImage = PIL.Image.fromarray(addition)
    if resize:
        resultImage = resizeimage.resize_thumbnail(resultImage, (256, 256))
    resultImage.save(os.path.join(original_image_annotation, output_name))

if __name__ == '__main__':

    # im1 = PIL.Image.open(os.path.join(data_dir, 'model_b', '00013_l_lip.png'))
    # im2 = PIL.Image.open(os.path.join(data_dir, 'model_b', '00013_u_lip.png'))
    # sumImage(im1, im2, resize=True, output_name='mouth_resize.png')

    resize_file(os.path.join(data_dir, 'model_b', '00013_nose.png'), os.path.join(data_dir, 'model_b', '00013_nose_resize.png'), (256, 256))
