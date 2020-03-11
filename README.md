# Neural Network Graph Homology Package

This package provides functionality for computing a graphical representation of
feedforward neural networks in pytorch. One may also use this package to compute
persistent homology of this graphical representation using the package. More
information on such an approach can be found in: https://arxiv.org/abs/1901.09496.

### Examples

Example Jupyter notebook scripts can be found in `/notebooks`.

### Dependencies

  - numpy
  - scipy
  - networkx
  - torch
  - torchvision

And if you want the package to compute homology for you (a la ripser.scikit-tda.org):

  - Cython
  - Ripser

To install, run

`$ pip install .` or if you want to edit the package `$ pip install -e .`.



### Limitations

This package is very opinionated on the model format and the types of neural
networks that it can work with. It is this opinionated because pytorch offers a
lot of flexibility in network model and module design, but changes to a network
model or module almost always affect the network graph. Capturing the network
graph for all possible pytorch options would be a monumental effort. Instead,
we focus on a few popular layer types that should provide the majority of
functionality.

Pytorch layer types currently implemented are:

  - Conv2d
    - we require dilation=1, groups=1, padding_mode='zeros'
  - MaxPool2d
    - we require dilation=1
  - Linear

We currently do not track bias weights as part of the network graph. This
functionality will be added in the future. For now, using networks without
bias parameters is recommended (although not required).

The max pooling operation is not well-defined in the case of parameter graphs.
For max pooling layers in the parameter graph, the current approach is to pass
forward the maximum weight of the previous layer's parameters for a given node.  
