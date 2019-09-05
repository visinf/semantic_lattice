import mxnet as mx
from mxnet.gluon import nn
import mxnet.ndarray as nd
import numpy as np
import skimage

from semantic_lattice.layers import permutohedral_lattice


class UpsamplingNetwork(mx.gluon.Block):
    """Implements architecture applicable to upsampling tasks."""

    def __init__(self,
                 task,
                 learning_mode,
                 num_data,
                 num_convolutions,
                 neighborhood_size,
                 feature_generator,
                 permutohedral_normalization,
                 dimension=1,
                 num_channels_embedding=64,
                 num_layers_embedding=5,
                 data_mean=None,
                 init_spatial_factor=1.,
                 init_intensity_factor=1.,
                 init_weight_factor=1.):
        super(UpsamplingNetwork, self).__init__()
        with self.name_scope():
            if task not in ["colorization", "flow", "segmentation"]:
                raise Exception("Task '%s' of UpsamplingNetwork is "
                                "unknown." % task)
            self.task = task
            self.num_data = num_data
            self.num_convolutions = num_convolutions
            self.neighborhood_size = neighborhood_size
            self.dimension = dimension
            self.feature_generator = feature_generator
            if learning_mode not in [
                    "scale_factors", "features_only", "weights_only",
                    "learn_all"
            ]:
                raise Exception("Learning mode '%s' of UpsamplingNetwork is "
                                "unknown." % learning_mode)
            if learning_mode in ["scale_factors", "weights_only"]:
                self.embedding = nn.Sequential()
                self.batchnorm = nn.Sequential()
            else:
                self.embedding = CnnEmbedding(dimension - 2,
                                              num_channels_embedding,
                                              num_layers_embedding)
                self.batchnorm = nn.BatchNorm(axis=1, center=True, scale=True)
            self.feature_factor_spatial = self.params.get(
                'feature_factor_spatial',
                shape=(1, ),
                init=mx.init.Constant(init_spatial_factor),
                allow_deferred_init=False)
            self.feature_factor_intensity = self.params.get(
                'feature_factor_intensity',
                shape=(1, ),
                init=mx.init.Constant(init_intensity_factor),
                allow_deferred_init=False)
            if learning_mode in ["features_only", "weights_only", "learn_all"]:
                self.feature_factor_spatial.__setattr__('grad_req', 'null')
                self.feature_factor_intensity.__setattr__('grad_req', 'null')
            self.weight_factor = None
            if learning_mode in ["scale_factors", "features_only"]:
                self.weight_factor = self.params.get(
                    'weight_factor',
                    shape=(1, ),
                    init=mx.init.Constant(init_weight_factor),
                    allow_deferred_init=False)
            if learning_mode in ["features_only"]:
                self.weight_factor.__setattr__('grad_req', 'null')

            init_array = _get_gaussian_initialization(
                dimension, neighborhood_size, num_data)
            initializer = mx.initializer.Constant(init_array)

            normalization_block = None
            if permutohedral_normalization == "end_to_end":
                normalization_block = self._stack_convolutional_layers(
                    initializer, freeze_weights=False, positive_weights=True)
            elif permutohedral_normalization == "end_to_end_fixed":
                normalization_block = self._stack_convolutional_layers(
                    initializer, freeze_weights=True, positive_weights=False)

            freeze_weights = (learning_mode in [
                "scale_factors", "features_only"
            ])
            convolution_block = self._stack_convolutional_layers(
                initializer, freeze_weights=freeze_weights)
            self.convolutions = permutohedral_lattice.PermutohedralBlock(
                neighborhood_size=neighborhood_size,
                lattice_size=None,
                convolution_block=convolution_block,
                normalization_type=permutohedral_normalization,
                normalization_block=normalization_block)
            self.data_mean = data_mean
            if data_mean is not None:
                self.guidance_mean, self.data_mean = data_mean

    def forward(self, data):
        """Returns the output of a forward pass for colorization."""
        if self.task == "colorization":
            return self.forward_colorization(data)
        elif self.task in ["flow", "segmentation"]:
            return self.forward_dense_prediction(data)

    def forward_colorization(self, data):
        """Returns the output of a forward pass for colorization."""
        guidance_small, guidance_large, data_small = data
        height, width = guidance_large.shape[2:]

        # Upsample data_small and guidance_small with nearest neighbors.
        # Use skimage as mx.nd.UpSampling allows only for one scale factor.
        batch_size = guidance_large.shape[0]
        data_small_upsampled = nd.zeros(
            (batch_size, data_small.shape[1], height, width),
            ctx=data_small.context)
        for batch_num in range(batch_size):
            data_help = skimage.transform.resize(
                np.transpose(data_small[batch_num].asnumpy(), (1, 2, 0)),
                (height, width),
                order=0,
                anti_aliasing=False,
                mode='constant')
            data_small_upsampled[batch_num] = nd.array(
                np.transpose(data_help, (2, 0, 1)), ctx=data_small.context)
        guidance_small_upsampled = nd.zeros(
            (batch_size, guidance_small.shape[1], height, width),
            ctx=data_small.context)
        for batch_num in range(batch_size):
            guidance_help = skimage.transform.resize(
                np.transpose(guidance_small[batch_num].asnumpy(), (1, 2, 0)),
                (height, width),
                order=0,
                anti_aliasing=False,
                mode='constant')
            guidance_small_upsampled[batch_num] = nd.array(
                np.transpose(guidance_help, (2, 0, 1)), ctx=data_small.context)

        # Generate features for small input data.
        with data_small.context:
            features_small = self.feature_generator(guidance_small_upsampled)
        # Scale spatial features by height/width to get invariance to size.
        with data_small.context:
            spatial_scaling = nd.array([[[[width]], [[height]]]])
        spatial_features = spatial_scaling * self.feature_factor_spatial.data(
        ) * features_small[:, :2]
        # Center remaining features if applicable.
        remaining_features = features_small[:, 2:]
        if self.data_mean is not None:
            remaining_features = remaining_features - self.guidance_mean.copyto(
                guidance_large.context)
        # Scale remaining feature and pass through embedding network.
        remaining_features = self.feature_factor_intensity.data(
        ) * remaining_features
        remaining_features = self.embedding(remaining_features)
        remaining_features = self.batchnorm(remaining_features)
        # Concatenate and reshape features_small.
        features_small = nd.concat(spatial_features, remaining_features, dim=1)
        features_small = features_small.reshape([0, 0, -1])

        # Generate features for large output data.
        with data_small.context:
            features_large = self.feature_generator(guidance_large)
        # Scale spatial features by height/width to get invariance to size.
        spatial_features = spatial_scaling * self.feature_factor_spatial.data(
        ) * features_large[:, :2]
        # Center remaining features if applicable.
        remaining_features = features_large[:, 2:]
        if self.data_mean is not None:
            remaining_features = remaining_features - self.guidance_mean.copyto(
                guidance_large.context)
        # Center remaining features and pass through embedding network.
        remaining_features = self.feature_factor_intensity.data(
        ) * remaining_features
        remaining_features = self.embedding(remaining_features)
        remaining_features = self.batchnorm(remaining_features)
        # Concatenate and reshape features_large.
        features_large = nd.concat(spatial_features, remaining_features, dim=1)
        features_large = features_large.reshape([0, 0, -1])

        # Concatenate input and output features.
        features = nd.concat(features_small, features_large, dim=2)
        features_in_size = features_small.shape[-1]
        features_out_size = features_large.shape[-1]

        # Reshape input data and guidance images.
        data_small_upsampled = data_small_upsampled.reshape([0, 0, -1])
        guidance_small_upsampled = guidance_small_upsampled.reshape([0, 0, -1])
        guidance_large = guidance_large.reshape([0, 0, -1])

        # Compute offset between data_small and guidance_small
        offset_small = data_small_upsampled - guidance_small_upsampled
        # Pass offset_small through permutohedral convolutions.
        offset_large = self.convolutions(offset_small, features, 0,
                                         features_in_size, features_in_size,
                                         features_out_size, self.weight_factor)
        # Generate output data from estimated offset.
        data_large = offset_large + guidance_large

        return data_large.reshape([0, 0, height, width])

    def forward_dense_prediction(self, data):
        """Returns the output of a forward pass for dense prediction tasks."""
        _, guidance_large, data_small = data
        height, width = guidance_large.shape[2:]

        # Upsample data_small with nearest neighbors if applicable.
        if data_small.shape[2:] == guidance_large.shape[2:]:
            data_small_upsampled = data_small
        else:
            # Use skimage as mx.nd.UpSampling allows only for one scale factor.
            batch_size = guidance_large.shape[0]
            data_small_upsampled = nd.zeros(
                (batch_size, data_small.shape[1], height, width),
                ctx=data_small.context)
            for batch_num in range(batch_size):
                data_help = skimage.transform.resize(
                    np.transpose(data_small[batch_num].asnumpy(), (1, 2, 0)),
                    (height, width),
                    order=0,
                    anti_aliasing=False,
                    mode='constant')
                data_small_upsampled[batch_num] = nd.array(
                    np.transpose(data_help, (2, 0, 1)), ctx=data_small.context)

        # Generate features.
        with data_small.context:
            features = self.feature_generator(guidance_large)
        # Scale spatial features by height/width to get invariance to size.
        with data_small.context:
            spatial_scaling = nd.array([[[[width]], [[height]]]])
        spatial_features = spatial_scaling * self.feature_factor_spatial.data(
        ) * features[:, :2]
        # Center remaining features if applicable.
        remaining_features = features[:, 2:]
        if self.data_mean is not None:
            remaining_features = (
                remaining_features - self.guidance_mean.copyto(
                    guidance_large.context))
        # Scale remaining features and pass through embedding network.
        remaining_features = self.feature_factor_intensity.data(
        ) * remaining_features
        remaining_features = self.embedding(remaining_features)
        remaining_features = self.batchnorm(remaining_features)

        # Concatenate and reshape features.
        features = nd.concat(spatial_features, remaining_features, dim=1)
        features = features.reshape([0, 0, -1])
        features_size = features.shape[-1]

        # Center input data if applicable.
        if self.data_mean is not None:
            data_small_upsampled = data_small_upsampled - self.data_mean.copyto(
                data_small.context)
        # Reshape input data.
        data_small_upsampled = data_small_upsampled.reshape([0, 0, -1])
        # Pass small data through permutohedral convolution.
        data_large = self.convolutions(data_small_upsampled, features, 0,
                                       features_size, 0, features_size,
                                       self.weight_factor)
        # Reshape output data.
        data_large = data_large.reshape([0, 0, height, width])
        # Revert centering if applicable.
        if self.data_mean is not None:
            data_large = data_large + self.data_mean.copyto(data_large.context)

        return data_large

    def _stack_convolutional_layers(self,
                                    initializer,
                                    freeze_weights=False,
                                    positive_weights=False):
        """Stacks several permutohedral convolution blocks."""
        convolution_block = permutohedral_lattice.PermutohedralSequential()
        for _ in range(self.num_convolutions - 1):
            convolution_block.add(
                permutohedral_lattice.PermutohedralConvolution(
                    self.num_data,
                    self.num_data,
                    self.dimension,
                    neighborhood_size=self.neighborhood_size,
                    groups=self.num_data,
                    weight_initializer=initializer,
                    activation='relu',
                    freeze_weight=freeze_weights,
                    positive_weight=positive_weights))
        # No ReLu for last convolution due to zero mean of data.
        if self.num_convolutions > 0:
            convolution_block.add(
                permutohedral_lattice.PermutohedralConvolution(
                    self.num_data,
                    self.num_data,
                    self.dimension,
                    neighborhood_size=self.neighborhood_size,
                    groups=self.num_data,
                    weight_initializer=initializer,
                    freeze_weight=freeze_weights,
                    positive_weight=positive_weights))
        return convolution_block


class CnnEmbedding(mx.gluon.Block):
    """Implements basic CNN for embedding."""

    def __init__(self, dimension, num_channels, num_layers):
        super(CnnEmbedding, self).__init__()
        with self.name_scope():
            self.convolutions = nn.Sequential()
            for _ in range(num_layers - 1):
                self.convolutions.add(
                    nn.Conv2D(
                        channels=num_channels,
                        kernel_size=3,
                        padding=1,
                        weight_initializer=mx.init.Xavier()))
                self.convolutions.add(nn.LeakyReLU(0.2))
            # No ReLu for last convolution to allow for negative features.
            self.convolutions.add(
                nn.Conv2D(
                    channels=dimension,
                    kernel_size=3,
                    padding=1,
                    weight_initializer=mx.init.Xavier()))

    def forward(self, data):
        """Returns the output of a forward network pass."""
        return self.convolutions(data)


def _get_gaussian_initialization(num_features, neighborhood_size, num_data):
    """Initializes permutohedral filter as Gaussian kernel."""
    file_name = './experiments/gaussian_initializations/' \
                'gaussian_filter_neighborhood{}_features{}' \
                '.npy'.format(neighborhood_size, num_features)
    init_array = nd.array(np.load(file_name))
    # Normalize filter for better initialization.
    init_array = init_array / nd.sum(init_array)
    return init_array.repeat(repeats=num_data, axis=0)
