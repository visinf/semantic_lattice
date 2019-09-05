# Semantic lattice

This source code release accompanies the paper

**Learning Task-Specific Generalized Convolutions in the Permutohedral
Lattice** \
Anne S. Wannenwetsch, Martin Kiefel, Peter Gehler, Stefan Roth.
In GCPR 2019.

The code provided in this repository targets the training of networks
for joint upsampling operations and illustrates the application of the
semantic lattice to the tasks of color, optical flow and
semantic segmentation upsampling.

Contact: Anne Wannenwetsch (anne.wannenwetsch@visinf.tu-darmstadt.de)

Requirements
------------
The code was tested with Python 3.5, MXNet 1.1 and Cuda 8.0.

Requirements for the semantic lattice can be installed with
```bash
pip install -r requirements.txt
```

To include the semantic lattice layer into MxNet,
the code needs to be compiled from source. Therefore, you should
checkout MXNet 1.1, e.g. by performing the following actions:
```bash
git clone https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
git checkout 1.1.0
```
Then, you can apply the provided patch for the semantic lattice
layer. For instance, you could do the following:
```bash
cp ~/semantic-lattice/src/semantic_lattice/layers/0001-Add-permutohedral-convolution.patch ~/incubator-mxnet
git am -3 < 0001-Add-permutohedral-convolution.patch
```
Please adapt the paths accordingly, if the directories `incubator-mxnet`
and `semantic-lattice` are not located in your home folder.

Prepare and start the compilation of MXNet, e.g. using
```bash
git submodule update --init --recursive
make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=<path/to/your/cuda8.0> USE_CUDNN=1
```

After compiling MXNet successfully, include the framework into your
environment which can be done by the following commands:
```bash
cd python
pip install -e .
```

Before running the code, make sure to set `PYTHONPATH` appropriately,
e.g. by performing the following:
```bash
cd ~/semantic-lattice
export PYTHONPATH=`pwd`/src
```

Training procedure
------------------
There are three different tasks for which code is provided:
* Color upsampling
* Optical flow upsampling
* Semantic segmentation upsampling

Depending on the specified options (see section below), the different 
networks are built. 
The training procedure is started as follows 
```bash
python bin/fit.py path/to/experiment/directory
```
and has several additional options:
* `--gpu <gpu_number>` Usage of gpu <gpu_number>.
* `--restore_checkpoint <name_of_checkpoint>` Specifies checkpoint from
which to start the training; must be included in experiment directory.
* `--start-epoch <epoch_number>` Start training from epoch <epoch_number>; 
important e.g. if a learning rate sheduler is used.
* `--evaluate` No training is performed but the network is evaluated on 
training, validation and test set; network to evaluate can be specified with 
`--restore_checkpoint`.

Test
----

For testing purposes, we have included test images from the Pascal VOC 
2012 and Sintel dataset as well as corresponding small data input for 
all tasks in `/test/data/<dataset_name>`.
To test the semantic lattice, please run 
```bash
python bin/fit.py ./experiments/config/colorization --restore_checkpoint checkpoint_both_learnt.params --gpu 0 --evaluate
python bin/fit.py ./experiments/config/optical_flow --restore_checkpoint checkpoint_both_learnt.params --gpu 0 --evaluate
python bin/fit.py ./experiments/config/semantic_segmentation --restore_checkpoint checkpoint_both_learnt.params --gpu 0 --evaluate
```
One should expect PSNR=35.37 for color upsampling, AEE=0.19 for 
optical flow upsampling and mIoU=0.99 for semantic segmentation 
upsampling on validation and test.

Please note that the train performance varies for multiple evaluation 
runs as the included option files specify that random cropping is
applied to training images.

Option files
------------
An option file `options.py` needs to be included in the experiment
directory and specifies all parameters necessary for the training
and test process. Moreover, the file is necessary to recreate the
network architecture for evaluation or resumption of training as MXNet
does not save the network structure. Sample files for the different
tasks are provided in `/experiments/config/<task_name>`.

Important parameters for upsampling networks
--------------------------------------------
* `num_dimensions`: Number of features used in the semantic lattice.
* `num_data`: Number of activations per pixel, i.e. data dimension.
* `num_convolutions`: Number of convolutions performed in semantic lattice.
* `neighborhood_size`: Size of permutohedral convolutions.
* `num_layers_embedding`: Number of  3x3 convolution layers in feature network.
* `num_channels_embedding`: Depth of convolution layers in feature network.
* `learning_mode`: Defines which elements of the network are learnt
during training. Please adjust permutohedral normalization type accordingly.
* `features`: List of features generated out of guidance data. Please 
provide spatial coordinates as first entry of the list.
* `initial_feature_factor_spatial`: Scale factor applied to spatial
features; should be determined by grid search.
* `initial_feature_factor_intensity`: Scale factor applied to remaining
features; should be determined by grid search.

For further parameters and the corresponding explanations, see the sample files 
`/experiments/config/<task_name>/options.py`.

Data input dense prediction tasks
---------------------------------

If the semantic lattice is applied for upsampling the results of dense 
prediction tasks, you need to provide low resolution outputs
of task-spefic networks. 
For our optical flow experiments, we used small sized flow estimates 
of `PWC-Net_ROB` and therefore downloaded the checkpoint `pwc_net.pth` 
from https://github.com/NVlabs/PWC-Net/tree/master/PyTorch.
For semantic segmentation, we applied a variante of DeepLabV3+ to generate 
low resolution segmentation maps. The corresponding checkpoint 
`xception_coco_voc_trainaug` can be found at 
https://github.com/qixuxiang/deeplabv3plus/blob/master/g3doc/model_zoo.md.

Estimates from task-specific networks have to be provided in a directory 
specified by the parameter `data_folder` in `options.py` and should have `
.npy` format. 
Samples files can be found in 
`/test/data/pascal/labels/small_predictions` and
`/test/data/Sintel/small_predictions`.
Please note that the semantic lattice takes small resolution network 
predictions as inputs, i.e. you should save the estimates _before_
the bilinear upsampling step.

Data splits
-----------

The data splits used in the paper for training, validation and test can
be found in `/experiments/lists/<task_name>`.

Pretrained networks
-------------------

We provide pretrained networks with learnt feature embeddings as well as
learnt permutohedral filters for all tasks in
`/experiments/config/<task_name>/checkpoint_both_learnt.params`.
As the name indicates, these networks correspond to the setting denoted
as `both learnt` in the paper.

Citation
--------

If you use our code, please cite our GCPR 2019 paper:

    @inproceedings{Wannenwetsch:2019:LTG,
        title = {Learning Task-Specific Generalized Convolutions in the
                 Permutohedral Lattice},
        author = {Anne S. Wannenwetsch and Martin Kiefel and
                  Peter V. Gehler and Stefan Roth},
        booktitle = {Pattern Recognition, 41st German Conference, GCPR 2019},
        year = {2019}}