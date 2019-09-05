import mxnet as mx
from mxnet import gluon
import numpy as np
from scipy.sparse import coo_matrix


def peak_signal_to_noise_ratio(max_value=1.):
    def compute_psnr(predictions, labels):
        mean_squared_error = gluon.loss.L2Loss(weight=2.)(predictions, labels)
        with mean_squared_error.context:
            max_value_array = mx.nd.array([max_value**2])
        return 10 * (
            mx.nd.log10(max_value_array) - mx.nd.log10(mean_squared_error))

    return compute_psnr


def softmax_cross_entropy_segmentation():
    def compute_cross_entropy_segmentation(predictions, labels):
        with labels.context:
            weight = (labels >= 0)
        loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)(predictions, labels,
                                                          weight)
        return loss * weight.size / mx.nd.sum(weight)

    return compute_cross_entropy_segmentation


def average_end_point_error():
    def compute_aepe(predictions, labels):
        labels, masks = labels
        epe = mx.nd.sqrt(mx.nd.sum((labels - predictions)**2, axis=1))
        epe = epe * masks[:, 0]
        aepe = mx.nd.sum(
            epe, axis=(1, 2)) / mx.nd.sum(
                masks[:, 0], axis=(1, 2))
        return mx.nd.mean(aepe)

    return compute_aepe


class LossMetric(object):
    """Customized evaluation metric for per instance loss functions."""

    def __init__(self, loss_function):
        self.loss_function = loss_function
        self.reset()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.number_samples = 0
        self.sum_loss = 0.0

    def update(self, predictions, labels):
        """Updates the loss metric."""
        current_loss = self.loss_function(predictions, labels)
        self.sum_loss += mx.nd.sum(current_loss).asscalar()
        self.number_samples += current_loss.shape[0]

    def get(self):
        """Gets the current evaluation result."""
        if self.number_samples == 0:
            return float('nan')
        return self.sum_loss / self.number_samples


class MiouMetric(object):
    """Mean intersection over union metric for semantic segmentation."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.confusion_matrix = 0

    def update(self, predictions, labels):
        """Updates the loss metric."""
        if predictions.shape[1] > 1:
            predictions = mx.nd.argmax(predictions, axis=1)
        else:
            predictions = mx.nd.clip(
                mx.nd.round(predictions), 0, self.num_classes - 1)
        labels = labels.reshape([-1]).asnumpy()
        predictions = predictions.reshape([-1]).asnumpy()
        valid_labels = (labels >= 0)
        self.confusion_matrix += _compute_confusion_matrix(
            predictions[valid_labels], labels[valid_labels], self.num_classes)

    def get(self):
        """Gets the current evaluation result."""
        sum_rows = np.sum(self.confusion_matrix, 0)
        sum_colums = np.sum(self.confusion_matrix, 1)
        diagonal_entries = np.diag(self.confusion_matrix)
        denominator = sum_rows + sum_colums - diagonal_entries

        valid_classes = (denominator != 0)
        num_valid_classes = np.sum(valid_classes)
        denominator += (1 - valid_classes)
        iou = diagonal_entries / denominator
        if num_valid_classes == 0:
            return float('nan')
        return np.sum(iou) / num_valid_classes


def _compute_confusion_matrix(predictions, labels, num_classes):
    if np.min(labels) < 0 or np.max(labels) >= num_classes:
        raise Exception("Labels out of bound.")

    if np.min(predictions) < 0 or np.max(predictions) >= num_classes:
        raise Exception("Predictions out of bound.")

    # Idea borrowed from tensorflow implementation
    values = np.ones(predictions.shape)
    confusion_matrix = coo_matrix((values, (labels, predictions)),
                                  shape=(num_classes, num_classes)).toarray()
    return confusion_matrix
