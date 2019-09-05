import mxnet as mx
from mxnet import gluon

from semantic_lattice import augmenters
from semantic_lattice import datasets
from semantic_lattice import losses
from semantic_lattice import upsampling_network
from semantic_lattice.layers import feature_generators


def dataset_from_parameters(params, evaluate=False):
    augmenter_train, augmenter_test = _augmenter(params)
    train_data, validation_data, test_data = _data(params, augmenter_train,
                                                   augmenter_test, evaluate)

    if train_data is None:
        raise Exception("Dataset '%s' is unknown!" % (params['dataset']))

    train_data_loader = gluon.data.DataLoader(
        train_data, params['batch_size'], shuffle=True, num_workers=24)
    validation_data_loader = gluon.data.DataLoader(
        validation_data,
        params['batch_size_evaluation'],
        shuffle=True,
        num_workers=24)

    if not evaluate:
        return train_data_loader, validation_data_loader

    test_data_loader = gluon.data.DataLoader(
        test_data, params['batch_size_evaluation'], shuffle=False)
    return train_data_loader, [validation_data_loader, test_data_loader]


def _augmenter(params):
    if params['dataset'] == "PascalColorization":
        return None, None
    elif params['dataset'] == "croppedPascalColorization":
        augmenter_train = augmenters.AugmenterColorization((200, 272))
        return augmenter_train, None
    elif params['dataset'] == "PascalSegmentation":
        augmenter = augmenters.AugmenterSegmentation((-1, -1),
                                                     original_shape=True)
        return augmenter, augmenter
    elif params['dataset'] == "croppedPascalSegmentation":
        augmenter_train = augmenters.AugmenterSegmentation((200, 272))
        augmenter_test = augmenters.AugmenterSegmentation((-1, -1),
                                                          original_shape=True)
        return augmenter_train, augmenter_test
    elif params['dataset'] == "Sintel":
        return None, None
    elif params['dataset'] == "croppedSintel":
        augmenter_train = augmenters.AugmenterFlow((281, 512))
        return augmenter_train, None
    raise Exception("Configuration '%s' is unknown!" % params['dataset'])


def _data(params, augmenter_train, augmenter_test, evaluate):
    if "PascalColorization" in params['dataset']:
        train_data = datasets.PascalColorization(
            params['train_data'],
            params['train_list'],
            params['downsampling_factor'],
            transform=augmenter_train)
        validation_data = datasets.PascalColorization(
            params['validation_data'],
            params['validation_list'],
            params['downsampling_factor'],
            transform=augmenter_test)
        test_data = None
        if evaluate:
            test_data = datasets.PascalColorization(
                params['test_data'],
                params['test_list'],
                params['downsampling_factor'],
                transform=augmenter_test)
        return train_data, validation_data, test_data
    elif "PascalSegmentation" in params['dataset']:
        train_data = datasets.PascalSegmentation(
            params['train_data'],
            params['train_list'],
            params['train_labels'],
            params['data_folder'],
            transform=augmenter_train)
        validation_data = datasets.PascalSegmentation(
            params['validation_data'],
            params['validation_list'],
            params['validation_labels'],
            params['data_folder'],
            transform=augmenter_test)
        test_data = None
        if evaluate:
            test_data = datasets.PascalSegmentation(
                params['test_data'],
                params['test_list'],
                params['test_labels'],
                params['data_folder'],
                transform=augmenter_test)
        return train_data, validation_data, test_data
    elif "Sintel" in params['dataset']:
        train_data = datasets.Sintel(
            params['train_data'],
            params['train_list'],
            params['train_labels'],
            params['data_folder'],
            transform=augmenter_train)
        validation_data = datasets.Sintel(
            params['validation_data'],
            params['validation_list'],
            params['validation_labels'],
            params['data_folder'],
            transform=augmenter_test)
        test_data = None
        if evaluate:
            test_data = datasets.Sintel(
                params['test_data'],
                params['test_list'],
                params['test_labels'],
                params['data_folder'],
                transform=augmenter_test)
        return train_data, validation_data, test_data
    return None


def network_from_parameters(params):
    if params['model'] == "upsamplingNetwork":
        feature_list = features_from_parameters(params, reshape=False)
        feature_generator = feature_generators.stack_features(feature_list)
        data_mean = datasets.get_dataset_mean(params['dataset'])
        return upsampling_network.UpsamplingNetwork(
            params["task"], params['learning_mode'], params['num_data'],
            params['num_convolutions'], params['neighborhood_size'],
            feature_generator, params['permutohedral_normalization'],
            params['num_dimensions'], params['num_channels_embedding'],
            params['num_layers_embedding'], data_mean,
            params['initial_spatial_factor'],
            params['initial_intensity_factor'],
            params['initial_weight_factor'])
    raise Exception("Model '%s' is unknown!" % (params['model']))


def features_from_parameters(params, reshape=True):
    feature_list = []
    for feature in params['features']:
        feature_name, feature_params = feature
        if feature_name == "spatial":
            feature_list.append(
                feature_generators.spatial_features(reshape, *feature_params))
        elif feature_name == "activations":
            feature_list.append(
                feature_generators.activation_features(reshape,
                                                       *feature_params))
        else:
            raise Exception("Feature '%s' is unknown!" % feature_name)
    return feature_list


def optimizer_settings_from_parameters(params, start_iteration):
    stop_factor_lr = params.get('stop_factor_lr', 1e-8)
    learning_rate_scheduler = mx.lr_scheduler.FactorScheduler(
        params['scheduler_step'],
        params['scheduler_factor'],
        stop_factor_lr=stop_factor_lr)

    optimizer_settings = dict()
    optimizer_settings['lr_scheduler'] = learning_rate_scheduler
    optimizer_settings['begin_num_update'] = start_iteration
    optimizer_settings['learning_rate'] = params['learning_rate']
    optimizer_settings['wd'] = params.get('weight_decay', 0.0)
    optimizer_settings['clip_gradient'] = params.get('gradient_clip', 0.0)

    if params['optimizer'] == 'adam':
        optimizer_settings['beta1'] = params.get('beta1', 0.9)
        optimizer_settings['beta2'] = params.get('beta1', 0.999)
    return optimizer_settings


def evaluation_criterion_from_parameters(params):
    if params['loss'] == "softmax_cross_entropy_segmentation":
        loss_function = losses.softmax_cross_entropy_segmentation()
    elif params['loss'] == "mean_squared_error":
        loss_function = gluon.loss.L2Loss(weight=2.)
    elif params['loss'] == "average_end_point_error":
        loss_function = losses.average_end_point_error()
    else:
        raise Exception("Loss function '%s' is unknown!" % params['loss'])

    if params['performance_criterion'] == "mean_squared_error":
        performance_metric = losses.LossMetric(gluon.loss.L2Loss(weight=2.))
    elif params['performance_criterion'] == "peak_signal_to_noise_ratio":
        performance_metric = losses.LossMetric(
            losses.peak_signal_to_noise_ratio())
    elif params['performance_criterion'] == "mean_intersection_over_union":
        performance_metric = losses.MiouMetric(params['num_classes'])
    elif params['performance_criterion'] == "average_end_point_error":
        performance_metric = losses.LossMetric(
            losses.average_end_point_error())
    else:
        raise Exception("Performance criterion '%s' is unknown!" %
                        params['performance_criterion'])

    return loss_function, performance_metric
