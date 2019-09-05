import os

import mxnet as mx
import numpy as np
from PIL import Image
import skimage


class PascalVOC(mx.gluon.data.dataset.Dataset):
    """Base implementation for Pascal VOC 2012 dataset."""

    def __init__(self, image_directory, image_list, transform=None):
        super(PascalVOC, self).__init__()
        self.image_directory = image_directory
        self.image_names = _read_image_list(image_list)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        raise NotImplementedError


class PascalColorization(PascalVOC):
    """Loads the Pascal VOC 2012 dataset for colorization."""

    def __init__(self,
                 image_directory,
                 image_list,
                 downsampling_factor,
                 transform=None):
        self.downsampling_factor = downsampling_factor
        super(PascalColorization, self).__init__(image_directory, image_list,
                                                 transform)

    def __getitem__(self, index):
        label_path = os.path.join(self.image_directory,
                                  self.image_names[index] + '.jpg')
        label = _load_image(label_path)

        if self.transform is not None:
            label, _ = self.transform(label, None)

        image_shape = label.shape
        data = skimage.transform.resize(
            label, (image_shape[0] // self.downsampling_factor,
                    image_shape[1] // self.downsampling_factor),
            anti_aliasing=False,
            mode='constant')

        label = _convert_image(label)
        data = _convert_image(data)
        guidance_small, guidance_large = self._get_guidance_images(data, label)

        return (guidance_small, guidance_large, data), label

    @staticmethod
    def _get_guidance_images(data, label):
        return _rgb2gray(data), _rgb2gray(label)


class PascalSegmentation(PascalVOC):
    """Loads the Pascal VOC 2012 dataset for segmentation."""

    def __init__(self,
                 image_directory,
                 image_list,
                 label_directory,
                 data_folder,
                 transform=None):
        self.label_directory = label_directory
        self.data_folder = data_folder
        super(PascalSegmentation, self).__init__(image_directory, image_list,
                                                 transform)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_directory,
                                  self.image_names[index] + '.jpg')
        guidance_large = _load_image(image_path)

        label_path = os.path.join(self.label_directory,
                                  self.image_names[index] + '.png')
        label = self._load_segmentation(label_path)

        data_path = os.path.join(self.label_directory, self.data_folder,
                                 self.image_names[index] + '.npy')
        data = np.load(data_path)

        if self.transform is not None:
            data, guidance_large, label = self.transform(
                data, guidance_large, label)

        label = _convert_image(label)
        data = _convert_image(data)

        guidance_large = _convert_image(guidance_large)

        return (guidance_large, guidance_large, data), label

    @staticmethod
    def _load_segmentation(segmentation_path):
        segmentation = Image.open(segmentation_path)
        segmentation = np.array(segmentation, dtype=np.float32)
        segmentation[segmentation == 255.] = -1
        return np.expand_dims(segmentation, axis=2)


class Sintel(mx.gluon.data.dataset.Dataset):
    """Loads the Sintel optical flow dataset."""

    def __init__(self,
                 image_directory,
                 flow_list,
                 flow_directory,
                 data_folder,
                 transform=None):
        self.flow_list = _read_image_list(flow_list)
        self.image_directory = image_directory
        self.flow_directory = flow_directory
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.flow_list)

    def __getitem__(self, index):
        flow_name = self.flow_list[index]
        file_name = flow_name[:-4]

        label_path = os.path.join(self.flow_directory, 'flow', flow_name)
        label = self.load_flo(label_path)

        image_path = os.path.join(self.image_directory, file_name + '.png')
        image = _load_image(image_path)

        mask_path = os.path.join(self.flow_directory, 'invalid',
                                 file_name + '.png')
        mask_flow = 1. - _load_image(mask_path)
        mask_flow = np.expand_dims(mask_flow, axis=2)

        data_path = os.path.join(self.data_folder, file_name + '.npy')
        data = np.load(data_path)

        if self.transform is not None:
            data, image, label, mask_flow = self.transform(
                data, image, label, mask_flow)

        label = _convert_image(label)
        mask_flow = _convert_image(mask_flow)
        data = _convert_image(data)
        guidance_large = _convert_image(image)

        return (guidance_large, guidance_large, data), (label, mask_flow)

    @staticmethod
    def load_flo(path):
        with open(path, 'rb') as flow_file:
            check_number = np.fromfile(flow_file, np.float32, count=1)
            if check_number != 202021.25:
                raise Exception("Invalid flow file at '%s'." % path)
            height = np.fromfile(flow_file, np.int32, count=1)[0]
            width = np.fromfile(flow_file, np.int32, count=1)[0]
            data = np.fromfile(flow_file, np.float32, count=2 * width * height)
        return np.resize(data, (width, height, 2))


def get_dataset_mean(dataset):
    if "PascalColorization" in dataset:
        data_mean = mx.nd.array([[[0.4000], [0.5314], [0.5368]]])
        guidance_mean = mx.nd.array([[[0.4927]]])
        return [guidance_mean, data_mean]
    if "PascalSegmentation" in dataset:
        data_mean = mx.nd.array([[[0.]]])
        guidance_mean = mx.nd.array([[[[0.4000]], [[0.5314]], [[0.5368]]]])
        return [guidance_mean, data_mean]
    if "Sintel" in dataset:
        data_mean = mx.nd.array([[[[-1.0210539]], [[1.3717544]]]])
        guidance_mean = mx.nd.array([[[[0.35742804]], [[0.32408935]],
                                      [[0.27271509]]]])
        return [guidance_mean, data_mean]

    raise Exception("Mean of dataset '%s' is unknown!" % dataset)


def _load_image(image_path):
    image = Image.open(image_path)
    return np.array(image) / 255.


def _convert_image(image):
    image = np.transpose(image, (2, 0, 1))
    return mx.nd.array(image)


def _read_image_list(image_list):
    with open(image_list) as list_file:
        image_names = list_file.read().splitlines()
    return image_names


def _rgb2gray(image):
    channel_factors = mx.nd.array([[[0.299]], [[0.587]], [[0.114]]])
    return mx.nd.sum(image * channel_factors, axis=0, keepdims=True)
