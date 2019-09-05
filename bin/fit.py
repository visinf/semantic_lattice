from importlib.machinery import SourceFileLoader
import logging
import os
import random
import sys

import click
import mxnet as mx
from mxnet import gluon
import numpy as np

from semantic_lattice import util

LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument(
    'experiment_directory',
    type=click.Path(exists=True, writable=True, file_okay=False))
@click.option('--restore_checkpoint', type=click.Path(dir_okay=False))
@click.option('--evaluate', is_flag=True)
@click.option('--start-epoch', type=int, default=0)
@click.option('--gpu', type=int, multiple=False, default=None)
def fit(experiment_directory, restore_checkpoint, evaluate, start_epoch, gpu):
    """Trains a model according to options in experiment_directory."""
    options_path = os.path.join(experiment_directory, 'options.py')
    options_file = SourceFileLoader('_options', options_path).load_module()
    LOGGER.info("Loading experiment configuration from `%s`", options_path)
    params = options_file.config

    random.seed(params['random_seed'])
    mx.random.seed(params['random_seed'])
    np.random.seed(params['random_seed'])

    if gpu is None:
        context = mx.cpu()
    else:
        context = mx.gpu(gpu)

    train_data, validation_data = util.dataset_from_parameters(
        params, evaluate)

    network = util.network_from_parameters(params)
    params_network = network.collect_params()
    if restore_checkpoint:
        checkpoint = os.path.join(experiment_directory, restore_checkpoint)
        params_network.load(checkpoint, ctx=context)
    else:
        params_network.initialize(mx.init.Xavier(), ctx=context)

    loss_function, performance_metric = \
        util.evaluation_criterion_from_parameters(params)

    if evaluate:
        validation_data, test_data = validation_data
        train_performance, train_loss = evaluate_network(
            train_data,
            network,
            context,
            performance_metric,
            loss_function=loss_function)
        validation_performance = evaluate_network(validation_data, network,
                                                  context, performance_metric)
        test_performance = evaluate_network(test_data, network, context,
                                            performance_metric)
        LOGGER.info('Evaluate %s:', restore_checkpoint)
        LOGGER.info(
            'Loss: %s, Train performance: %s, Validation performance: '
            '%s, Test performance: %s', train_loss, train_performance,
            validation_performance, test_performance)
        return

    # Adjust learning rate for permutohedral kernels if applicable.
    if params['learning_mode'] == "learn_all":
        for parameter in params_network.values():
            if 'permutohedral' in parameter.name:
                if parameter.grad_req is not 'null':
                    parameter.lr_mult = params[
                        'lr_factor_permutohedral_filters']

    # Prepare averaging over batches if applicable.
    batch_counter = 0
    averaged_batches = params.get('averaged_batches', 1.0)
    if averaged_batches > 1:
        for parameter in params_network.values():
            if parameter.grad_req is not 'null':
                parameter.grad_req = 'add'

    optimizer_settings = util.optimizer_settings_from_parameters(
        params,
        start_epoch * len(train_data) / averaged_batches)
    trainer = gluon.Trainer(params_network, params['optimizer'],
                            optimizer_settings)

    num_epochs = int(
        round(params['num_iterations'] * averaged_batches / len(train_data)))
    LOGGER.info("Number of epochs to be performed: %s", num_epochs)

    for epoch in range(start_epoch, num_epochs):
        for index, (image, label) in enumerate(train_data):
            image = _transfer_to_context(image, context)
            label = _transfer_to_context(label, context)

            with mx.autograd.record():
                output = network(image)
                loss = loss_function(output, label)
            loss.backward()

            batch_counter += 1
            if batch_counter == averaged_batches:
                trainer.step(averaged_batches * params['batch_size'])
                for parameter in params_network.values():
                    if parameter.grad_req is not 'null':
                        parameter.zero_grad()
                batch_counter = 0
            mx.nd.waitall()

        if (epoch + 1) % params['checkpoint_frequency'] == 0 or \
                epoch == num_epochs - 1:
            checkpoint_file = os.path.join(
                experiment_directory,
                'checkpoint_epoch{}.params'.format(epoch + 1))
            params_network.save(checkpoint_file)

            train_performance, train_loss = evaluate_network(
                train_data,
                network,
                context,
                performance_metric,
                loss_function=loss_function)
            validation_performance = evaluate_network(
                validation_data, network, context, performance_metric)
            LOGGER.info(
                'Epoch %s. Loss: %s, Train performance: %s, Validation '
                'performance: %s', epoch + 1, train_loss, train_performance,
                validation_performance)


def evaluate_network(data_iterator,
                     network,
                     context,
                     performance_metric,
                     loss_function=None):
    """Returns performance_metric and loss of network on data_iterator."""
    performance_metric.reset()
    loss = mx.metric.Loss()
    for index, (image, label) in enumerate(data_iterator):
        image = _transfer_to_context(image, context)
        label = _transfer_to_context(label, context)
        output = network(image)
        performance_metric.update(output, label)
        if loss_function is not None:
            loss.update(None, loss_function(output, label))
        mx.nd.waitall()
    if loss_function is not None:
        return performance_metric.get(), loss.get()[1]
    return performance_metric.get()


def _transfer_to_context(data, context):
    if isinstance(data, list):
        return [element.as_in_context(context) for element in data]
    return data.as_in_context(context)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    LOGGER.info('Command line arguments: ' + ' '.join(sys.argv))
    fit()
