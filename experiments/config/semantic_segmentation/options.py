config = {
    # Model. #######################
    # Supported are: "upsamplingNetwork".
    "model": "upsamplingNetwork",
    # Task to which upsamplingNetwork is applied, supported: "colorization",
    # "flow", "segmentation".
    "task": "segmentation",
    # Number of dimensions used in upsamplingNetwork.
    "num_dimensions": 5,
    # Number of activations per pixel in upsamplingNetwork.
    "num_data": 21,
    # Kind of normalization applied in permutohedral lattice.
    # None = no normalization applied.
    # "end_to_end" = division by output of applying splat, convolution and slice
    #       operations on all-1-array; convolution weight needs to be positive.
    # "end_to_end_fixed" = division by output of applying splat, convolution and
    #       slice operations on all-1-array; normalization weight is fixed.
    "permutohedral_normalization": "end_to_end",

    # Upsampling parameters. #######################
    # Number of convolutions performed in permutohedral lattice.
    "num_convolutions": 1,
    # Neighborhood size of convolutions performed in permutohedral lattice.
    "neighborhood_size": 1,
    # Depth of embedding networks.
    "num_channels_embedding": 15,
    # Number of layers in embedding networks.
    "num_layers_embedding": 3,
    # Learning mode of network,
    # "scale_factors" = scale factors learnt for features and weights,
    #                   use permutohedral_normalization = "end_to_end_fixed"
    #                   to avoid learning normalization weights.
    # "features_only" = only feature embedding learnt,
    #                   use permutohedral_normalization = "end_to_end_fixed"
    #                   to avoid learning normalization weights.
    # "weights_only" = only permutohedral convolution weights learnt.
    # "learn_all" = features and permutohedral weights both learnt.
    "learning_mode": "learn_all",
    # Initial parameter for scale factor of spatial features.
    "initial_spatial_factor": 0.15,
    # Initial parameter for scale factor of remaining features.
    "initial_intensity_factor": 25.,
    # Initial parameter for weight scale factor.
    "initial_weight_factor": 1.,

    # Feature components used for the upsampling network. Available: "spatial",
    # "activations". Feature parameters need to be included as a list in second
    # list element. Spatial features have to be the first entry of the feature
    # list due to processing of features in upsamplingNetwork.
    "features": [("spatial", [0]), ("activations", [1.0])],

    # Data and preprocessing. ########################
    # Used dataset, supported: "PascalColorization",
    # "croppedPascalColorization", "Sintel", "croppedSintel",
    # "PascalSegmenation", "croppedPascalSegmentation".
    "dataset": "croppedPascalSegmentation",
    # Downsampling factor applied to the dataset images, only necessary for
    # colorization task.
    "downsampling_factor": None,
    # Path to folder containing small predictions, i.e. outputs of dense
    # prediction networks before bilinear upsampling step, used for dense
    # prediction tasks only.
    "data_folder": "small_predictions",
    # Path to the list of samples used for training data.
    "train_list": './data/pascal/train_small.txt',
    # Path to the image directory used for training data.
    "train_data": './data/pascal/images',
    # Path to the label directory used for training data, only necessary for
    # dense prediction tasks.
    "train_labels": './data/pascal/labels',
    # Path to the list of samples used for validation data.
    "validation_list": './data/pascal/train_small.txt',
    # Path to the image directory used for validation data.
    "validation_data": './data/pascal/images',
    # Path to the label directory used for validation data, only necessary for
    # dense prediction tasks.
    "validation_labels": './data/pascal/labels',
    # Path to the list of samples used for test data.
    "test_list": './data/pascal/train_small.txt',
    # Path to the image directory used for test data.
    "test_data": './data/pascal/images',
    # Path to the label directory used for test data, only necessary for dense
    # prediction tasks.
    "test_labels": './data/pascal/labels',

    # Optimizer ####################
    "optimizer": "adam",
    # Number of optimization steps.
    "num_iterations": 8,
    # Batch size used during training.
    "batch_size": 2,
    # Batch size used during evaluation.
    "batch_size_evaluation": 1,
    # Number of batches averaged for gradient computation.
    "averaged_batches": 2,
    # Base learning rate.
    "learning_rate": 1e-4,
    # Additional multiplication factor for learning rates of permutohedral
    # filter weights. Only applicable for learning mode = "learn_all".
    "lr_factor_permutohedral_filters": 1e-4,
    # Amount of optimization steps after which the learning rate is cut.
    "scheduler_step": 2,
    # Ratio with which the learning rate is cut.
    "scheduler_factor": 0.1,
    # Value at which cutting of learning rate should be stopped.
    "stop_factor_lr" : 1e-5,
    # Parameter used for weight decay.
    "weight_decay": 0.0,
    # Threshold for gradient clipping of all parameters.
    "gradient_clip": 0.1,

    # Infrastructure. ##############
    # Loss used during training, choices: "mean_squared_error", \
    # "average_end_point_error", "softmax_cross_entropy_segmentation".
    "loss": "softmax_cross_entropy_segmentation",
    # Criterion evaluated to measure network performance, supported are
    # "mean_squared_error", "peak_signal_to_noise_ratio",
    # "average_end_point_error", "mean_intersection_over_union".
    "performance_criterion": "mean_intersection_over_union",
    # Number of possible segmentation classes, only necessary for
    # performance_criterion = "mean_intersection_over_union".
    "num_classes": 21,
    # After every x epochs, save and evaluate model.
    "checkpoint_frequency": 1,
    # Random seed used for initializations etc.
    "random_seed": 42,
}
