from skimage import transform
import torch
import numpy as np


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    @staticmethod
    def rescale_img(im, output_size):
        h, w = im.shape[:2]
        if isinstance(output_size, int):
            if h > w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size

        new_h, new_w = int(new_h), int(new_w)

        return transform.resize(im, (new_h, new_w), mode='constant')

    def __call__(self, sample):
        return {'code': sample['code'], 'image': self.rescale_img(sample['image'], self.output_size)}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    @staticmethod
    def crop_img(im, output_size):
        h, w = im.shape[:2]
        new_h, new_w = output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # top = 1
        # left = 1

        return im[top: top + new_h, left: left + new_w]

    def __call__(self, sample):
        return {'code': sample['code'], 'image': self.crop_img(sample['image'], self.output_size)}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    @staticmethod
    def to_tensor_img(im):
        if len(im.shape) == 3:
            return torch.from_numpy(im.transpose((2, 0, 1))).type(torch.FloatTensor)
        else:  # grayscale images
            # print 'grayscale img found. Size:', im.shape
            return torch.from_numpy(im).type(torch.FloatTensor).unsqueeze_(-1).expand(-1, -1, 3).permute(2, 0, 1)

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        try:
            res = {'code': sample['code'], 'image': self.to_tensor_img(sample['image'])}
        except ValueError as e:
            print sample['code']
            raise e

        return res
