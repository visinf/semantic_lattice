import numpy as np
import skimage


class AugmenterColorization(object):
    """Augments colorization by random cropping to size crop_shape."""

    def __init__(self, crop_shape):
        self.crop_shape = crop_shape

    def __call__(self, image, label):
        image_shape = image.shape
        if image_shape < self.crop_shape:
            raise Exception("Image too small for given crop_shape.")

        start_height = np.random.RandomState().randint(image_shape[0] -
                                                       self.crop_shape[0] + 1)
        end_height = start_height + self.crop_shape[0]
        start_width = np.random.RandomState().randint(image_shape[1] -
                                                      self.crop_shape[1] + 1)
        end_width = start_width + self.crop_shape[1]
        return image[start_height:end_height, start_width:end_width, :], label


class AugmenterFlow(object):
    """Augments flow data by random cropping to size crop_shape."""

    def __init__(self, crop_shape):
        self.crop_shape = crop_shape

    def __call__(self, data, image, label, mask_flow):
        label_shape = label.shape
        if label_shape < self.crop_shape:
            raise Exception("Label too small for given crop_shape.")

        start_height = np.random.RandomState().randint(label_shape[0] -
                                                       self.crop_shape[0] + 1)
        end_height = start_height + self.crop_shape[0]
        start_width = np.random.RandomState().randint(
            0, label_shape[1] - self.crop_shape[1] + 1)
        end_width = start_width + self.crop_shape[1]
        label = label[start_height:end_height, start_width:end_width, :]
        image = image[start_height:end_height, start_width:end_width, :]
        mask_flow = mask_flow[start_height:end_height, start_width:
                              end_width, :]

        data = skimage.transform.resize(
            data, (label_shape[0], label_shape[1]),
            anti_aliasing=False,
            mode='constant',
            order=0)
        data = data[start_height:end_height, start_width:end_width, :]
        return data, image, label, mask_flow


class AugmenterSegmentation(object):
    """Reshapes and if applicable augments segmentation data."""

    def __init__(self, crop_shape, original_shape=False):
        """Initializes the segmentation augmenter.
        Args:
            crop_shape: Size of cropped data.
            original_shape: If true no cropping but only reshaping is applied.
        """
        self.crop_shape = crop_shape
        self.original_shape = original_shape

    def __call__(self, data, image, label):
        label_shape = label.shape
        if self.original_shape:
            self.crop_shape = label_shape
        if label_shape < self.crop_shape:
            raise Exception("Label too small for given crop_shape.")

        if not self.original_shape:
            start_height = np.random.RandomState().randint(
                label_shape[0] - self.crop_shape[0] + 1)
            end_height = start_height + self.crop_shape[0]
            start_width = np.random.RandomState().randint(
                label_shape[1] - self.crop_shape[1] + 1)
            end_width = start_width + self.crop_shape[1]
            label = label[start_height:end_height, start_width:end_width, :]
            image = image[start_height:end_height, start_width:end_width, :]
        else:
            start_height = 0
            end_height = label_shape[0]
            start_width = 0
            end_width = label_shape[1]

        if data.shape[:2] != label_shape[:2]:
            data = skimage.transform.resize(
                data, (513, 513),
                anti_aliasing=False,
                mode='constant',
                order=0)
        data = data[start_height:end_height, start_width:end_width, :]
        return data, image, label
