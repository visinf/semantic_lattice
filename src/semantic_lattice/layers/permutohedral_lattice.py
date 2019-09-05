# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mxnet as mx
from mxnet import gluon
from mxnet import init
from mxnet import nd


class PermutohedralBlock(gluon.Block):
    """Permutohedral lattice block."""

    def __init__(self,
                 neighborhood_size=0,
                 lattice_size=None,
                 convolution_block=None,
                 normalization_type=None,
                 normalization_block=None):
        """Initializes the permutohedral lattice.

        Args:
            neighborhood_size: Extension of neighborhood considered in
                convolution.
            lattice_size: Number of lattice cells.
            convolution_block: Block of (stacked) permutohedral convolutions.
                If convolution_block is None, no convolution is performed.
            normalization_type: Normalization applied to output data.
            normalization_block: Block of permutohedral convolutions applied
                during normalization. If normalization_block is None,
                convolution_block is applied. Weights of normalization_block
                need to remain positive for a valid normalization operation.
        """
        super(PermutohedralBlock, self).__init__()
        with self.name_scope():
            self.neighborhood_size = neighborhood_size
            self.lattice_size = lattice_size
            self.convolution_block = convolution_block
            if normalization_type not in [
                    None, "end_to_end", "end_to_end_fixed"
            ]:
                raise Exception("Permutohedral normalization type '%s' is "
                                "unknown!" % normalization_type)
            self.normalization_type = normalization_type
            self.normalization_block = normalization_block
            if self.normalization_block is None:
                self.normalization_block = convolution_block

    def forward(self,
                data,
                features,
                features_in_offset,
                features_in_size,
                features_out_offset,
                features_out_size,
                weight_factor=None):
        """Returns convolved data at locations defined by features.

        Args:
            features: Input and output features concatenated along dimension=2.
            data: Input data corresponding to input pixels.
            features_in_offset: Offset of input feature positions in features.
            features_in_size: Number of input pixels.
            features_out_offset: Offset of output feature positions in features.
            features_out_size: Number of output pixels.
            weight_factor: Optional parameter that scales convolution weights.
        """

        if self.lattice_size is None:
            self.lattice_size = (features_in_size + features_out_size) * 10

        current_context = data.context
        if current_context != mx.cpu():
            # Push features and data to cpu as there is no gpu implementation
            # for lattice and splat.
            features = features.copyto(mx.cpu())
            data = data.copyto(mx.cpu())

        barycentric, offset, blur_neighbors = nd.permutohedral_lattice(
            nd.swapaxes(features, 1, 2),
            neighborhood_size=self.neighborhood_size,
            lattice_size=self.lattice_size)
        data_lattice = nd.permutohedral_splat(
            data,
            barycentric,
            offset,
            features_in_offset=features_in_offset,
            features_in_size=features_in_size,
            lattice_size=self.lattice_size)

        if self.convolution_block is None:
            data_blurred = data_lattice
        else:
            if current_context != mx.cpu():
                # Push data and blur_neighbors to gpu to perform convolution.
                data_lattice = data_lattice.copyto(current_context)
                blur_neighbors = blur_neighbors.copyto(current_context)

            data_blurred = self.convolution_block(data_lattice, blur_neighbors,
                                                  weight_factor)

        if current_context != mx.cpu():
            # Push data to cpu as there is no gpu implementation of slice.
            data_blurred = data_blurred.copyto(mx.cpu())

        data_out = nd.permutohedral_slice(
            data_blurred,
            barycentric,
            offset,
            features_out_offset=features_out_offset,
            features_out_size=features_out_size)

        if self.normalization_type is not None:
            normalization = get_normalization(
                barycentric, offset, blur_neighbors, self.normalization_block,
                weight_factor, features_in_offset, features_in_size,
                features_out_offset, features_out_size, self.lattice_size,
                data_lattice.shape[1])
            # Set normalization value to 1 for zero entries.
            normalization = normalization + (normalization == 0)
            data_out = data_out / normalization

        if current_context != mx.cpu():
            # Push output data back to original context.
            data_out = data_out.copyto(current_context)

        return data_out


class PermutohedralConvolution(gluon.Block):
    """Permutohedral lattice convolution."""

    def __init__(self,
                 num_filter,
                 num_data,
                 num_features,
                 neighborhood_size=0,
                 groups=1,
                 weight_initializer=init.Xavier(),
                 activation=None,
                 freeze_weight=False,
                 positive_weight=False):
        """Initializes the permutohedral convolution.

        Args:
            num_filter: Number of filters to learn.
            num_data: Number of data points per pixel.
            num_features: Number of features per pixel.
            neighborhood_size: Extension of neighborhood considered in
                convolution.
            groups: Number of groups into which the input data is sliced.
            weight_initializer: Initializer for convolution weights.
            activation: Activation function applied after convolution.
            freeze_weight: Flag to fix weight during training process.
            positive_weight: Flag to restrict weight to positivity.
        """
        super(PermutohedralConvolution, self).__init__()
        with self.name_scope():
            self.num_filter = num_filter
            self.neighborhood_size = neighborhood_size
            self.groups = groups
            self.positive_weight = positive_weight
            if positive_weight:
                weight_initializer = _LogInitializer(weight_initializer)
            self.weight = self.params.get(
                'weight',
                shape=(groups, int(num_filter / groups),
                       int(num_data / groups * _get_filter_size(
                           neighborhood_size, num_features))),
                init=weight_initializer,
                allow_deferred_init=False)
            if freeze_weight:
                self.weight.__setattr__('grad_req', 'null')
            self.activation = None
            if activation is not None:
                self.activation = gluon.nn.Activation(activation)

    def forward(self, data, blur_neighbors, weight_factor=None):
        weight = self.weight.data()
        if self.positive_weight:
            weight = nd.exp(weight)
        if weight_factor is not None:
            weight = weight_factor.data() * weight
        blurred_data = nd.permutohedral_convolve(
            data,
            weight,
            blur_neighbors,
            num_filter=self.num_filter,
            groups=self.groups)
        if self.activation is not None:
            blurred_data = self.activation(blurred_data)
        return blurred_data


class PermutohedralSequential(gluon.nn.Sequential):
    """Stacks permutohedral convolution blocks sequentially.

    Follows implementation of nn.Sequential and extents it to the
    multiple inputs required for permutohedral convolutions.
    """

    def __init__(self, prefix=None, params=None):
        super(PermutohedralSequential, self).__init__(
            prefix=prefix, params=params)

    def forward(self, data, blur_neighbors, weight_factor=None):
        for block in self._children:
            data = block(data, blur_neighbors, weight_factor)
        return data


def _get_filter_size(neighborhood_size, feature_size):
    """Returns number of elements in filter with size neighborhood_size."""
    return (neighborhood_size + 1)**(feature_size + 1) - neighborhood_size**(
        feature_size + 1)


def get_normalization(barycentric, offset, blur_neighbors, convolution_block,
                      weight_factor, features_in_offset, features_in_size,
                      features_out_offset, features_out_size, lattice_size,
                      num_data):
    """Returns normalization for lattice convolution with given features."""
    batch_size = barycentric.shape[0]
    dummy_data_in = nd.ones((batch_size, num_data, features_in_size))
    normalization = nd.permutohedral_splat(
        dummy_data_in,
        barycentric,
        offset,
        features_in_offset=features_in_offset,
        features_in_size=features_in_size,
        lattice_size=lattice_size)
    if convolution_block is not None:
        current_context = blur_neighbors.context
        if current_context != mx.cpu():
            normalization = normalization.copyto(current_context)
        normalization = convolution_block(normalization, blur_neighbors,
                                          weight_factor)
        if current_context != mx.cpu():
            normalization = normalization.copyto(mx.cpu())
    return nd.permutohedral_slice(
        normalization,
        barycentric,
        offset,
        features_out_offset=features_out_offset,
        features_out_size=features_out_size)


@mx.initializer.register
class _LogInitializer(mx.initializer.Initializer):
    def __init__(self, initializer):
        super(_LogInitializer, self).__init__()
        self.base_initializer = initializer

    def _init_weight(self, layer_name, layer_value):
        # Fix necessary as _init_bias is not called correctly.
        if layer_name.endswith('bias'):
            self._init_bias(layer_name, layer_value)
        else:
            # Call private function of base_initializer to bypass standard
            # mxnet behaviour which results in an infinite recursion.
            self.base_initializer._init_weight(layer_name, layer_value)
            layer_value[:] = nd.log(layer_value[:])

    def _init_bias(self, layer_name, layer_value):
        raise Exception('Bias initialization not implemented for '
                        '_LogInitializer')
