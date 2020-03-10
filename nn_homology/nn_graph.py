import collections
from functools import partial

import numpy as np
import networkx as nx
import scipy.sparse
import torch

def inverse_abs(x):
    return np.abs(1/x)

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape

    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    ''' An implementation of im2col based on some fancy indexing '''
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    '''An implementation of col2im based on fancy indexing and np.add.at'''
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

def conv_layer_as_matrix(X, X_names, W, stride, padding):
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    X_names_col = im2col_indices(X_names, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col
    out = out.reshape(n_filters, int(h_out), int(w_out), n_x)
    out = out.transpose(3, 0, 1, 2)

    return out, X_col, W_col, X_names_col

def add_conv(G, input_size, p, name_this, name_next, stride, padding, weight_func=inverse_abs, next_linear=False, X=None):
    '''adds convolutional layer to graph and returns updated graph'''
    conv_format = '{}_{}_{}'
    input_channels = p.shape[1]
    X = np.ones(input_size) if X is None else X
    for c in range(input_channels):
        print('Channel: {}'.format(c))
        X_names = np.arange(X.shape[2]*X.shape[3]).reshape((1,1,X.shape[2],X.shape[3]))
        tx = X[:,c,:,:].reshape((X.shape[0],1,X.shape[2],X.shape[3]))
        # convert to matrix information
        mat, X_col, W_col, xnames = conv_layer_as_matrix(tx,X_names,p[:,c,:,:].reshape((p.shape[0],1,p.shape[2],p.shape[3])),stride,padding)
        for f in range(W_col.shape[0]):
            for row in range(X_col.shape[0]):
                for col in range(X_col.shape[1]):
                    v = W_col[f,row] * X_col[row,col]
                    if v != 0:
                        if next_linear:
                            # next layer is linear
                            G.add_edge(conv_format.format(name_this,c,xnames[row,col]),conv_format.format(name_next,0,int((X_col.shape[1]*c) + (f*X_col.shape[1]) + col)), weight=weight_func(v))
                        else:
                            # next layer is conv
                            G.add_edge(conv_format.format(name_this,c,xnames[row,col]),conv_format.format(name_next,f,col), weight=weight_func(v))
    input_size = mat.shape
    return G, input_size

def add_mp(G, input_size, name_this, name_next, kernel_size, stride, padding, next_linear=False):
    '''adds max pooling layer to graph and returns updated graph'''
    conv_format = '{}_{}_{}'
    p = np.ones((input_size[0],input_size[1],kernel_size[0],kernel_size[1]))
    input_channels = p.shape[1]
    # next layer also conv
    X = np.ones(input_size)
    for c in range(input_channels):
        print('Channel: {}'.format(c))
        X_names = np.arange(X.shape[2]*X.shape[3]).reshape((1,1,X.shape[2],X.shape[3]))
        tx = X[:,c,:,:].reshape((X.shape[0],1,X.shape[2],X.shape[3]))
        # convert to matrix information
        mat, X_col, W_col, xnames = conv_layer_as_matrix(tx,X_names,p[:,c,:,:].reshape((p.shape[0],1,p.shape[2],p.shape[3])),stride,padding)
        for f in range(W_col.shape[0]):
            for row in range(X_col.shape[0]):
                for col in range(X_col.shape[1]):
                    node_name = conv_format.format(name_this,c,xnames[row,col])
                    ews = sorted(G.in_edges(node_name, data=True), key=lambda x: x[2]['weight'])
                    if len(ews) > 0:
                        v = ews[0][2]['weight']
                        if v != 0:
                            if next_linear:
                                G.add_edge(conv_format.format(name_this,c,xnames[row,col]),conv_format.format(name_next,0,int((X_col.shape[1]*c) + (f*X_col.shape[1]) + col)), weight=v)
                            else:
                                G.add_edge(conv_format.format(name_this,c,xnames[row,col]),conv_format.format(name_next,f,col), weight=v)
    input_size = [mat.shape[0], input_channels, mat.shape[2], mat.shape[3]]
    return G, input_size

def add_mp_act(G, X, name_this, name_next, kernel_size, stride, padding, weight_func=inverse_abs, next_linear=False):
    '''adds max pooling layer to graph and returns updated graph'''
    conv_format = '{}_{}_{}'
    p = np.ones((X.shape[0],X.shape[1],kernel_size[0],kernel_size[1]))
    input_channels = p.shape[1]
    # next layer also conv
    for c in range(input_channels):
        print('Channel: {}'.format(c))
        X_names = np.arange(X.shape[2]*X.shape[3]).reshape((1,1,X.shape[2],X.shape[3]))
        tx = X[:,c,:,:].reshape((X.shape[0],1,X.shape[2],X.shape[3]))
        # convert to matrix information
        mat, X_col, W_col, xnames = conv_layer_as_matrix(tx,X_names,p[:,c,:,:].reshape((p.shape[0],1,p.shape[2],p.shape[3])),stride,padding)
        for f in range(W_col.shape[0]):
            for row in range(X_col.shape[0]):
                for col in range(X_col.shape[1]):
                    v = W_col[f,row] * X_col[row,col]
                    if v != 0:
                        if next_linear:
                            # next layer is linear
                            G.add_edge(conv_format.format(name_this,c,xnames[row,col]),conv_format.format(name_next,0,int((X_col.shape[1]*c) + (f*X_col.shape[1]) + col)), weight=weight_func(v))
                        else:
                            # next layer is conv
                            G.add_edge(conv_format.format(name_this,c,xnames[row,col]),conv_format.format(name_next,f,col), weight=weight_func(v))
    input_size = [mat.shape[0], input_channels, mat.shape[2], mat.shape[3]]
    return G, input_size

def add_linear_linear(G, p, name_this, name_next, weight_func=inverse_abs, X=None):
    '''Adds linear layer to graph and returns updated graph. If X is given,
    compute the activation graph, otherwise compute the parameter graph. This
    function creates a graphical representation of the matrix multiply operation.

    Args:
        G (networkx.Graph): the networkx representation of the graph to which to
            add the linear layer.
        p (numpy.array): a numpy representation of the parameters of the layer
            of dimensionality [|l+1|,|l|] where |l| represents the number of
            nodes in layer l and |l+1| represents the number of nodes in the
            following layer.
        name_this (str): the name of layer l.
        name_next (str): the name of layer l+1.
        weight_func (callable, optional): a function to apply to the weight values.
            Should take a single float-like argument and return a float-like argument.
        X (numpy.array, optional): the activation values at each node in layer l.
    '''
    conv_format = '{}_{}_{}'
    p = p if X is None else p*X
    for row in range(p.shape[1]):
        for col in range(p.shape[0]):
            v = p[col,row]
            if v != 0:
                G.add_edge(conv_format.format(name_this,0,row),conv_format.format(name_next,0,col), weight=weight_func(v))
    return G

def to_directed_networkx(params, input_size):
    '''Create networkx representation of parameter graph of neural network. This
    function takes a list of parameter values and a list of activation values,
    both in the form of a list of numpy arrays (converted from pytorch tensor),
    one element in the list per layer.

    Args:
        params (list[numpy.array]): the weight (parameter) values of the network
            at each layer.
        input_size (tuple): the size of the first layer's input in form
            (batch, channels, heigh, width).

    Returns:
        networkx.DiGraph: The networkx representation of the activation network.
    '''
    # store all network info here
    G = nx.DiGraph()

    # assume last layer linear, loop over each layer and process
    for l in range(len(params)-1):

        # get parameter information for current layer
        param = params[l]
        # need to look ahead at next layer to get naming correct
        param_next = params[l+1]

        print('Layer: {}'.format(param['name']))

        # check the layer type to decide how to process
        if param['layer_type'] == 'Conv2d':

            if param_next['layer_type'] == 'Conv2d' or param_next['layer_type'] == 'MaxPool2d':
                # add edges and nodes of this layer to the networkx representation
                G, input_size = add_conv(G, input_size, param['param'], param['name'], param_next['name'], param['stride'], param['padding'], next_linear=False)

            elif param_next['layer_type'] == 'Linear':

                G, input_size = add_conv(G, input_size, param['param'], param['name'], param_next['name'], param['stride'], param['padding'], next_linear=True)

        elif param['layer_type'] == 'MaxPool2d':

            if param_next['layer_type'] == 'Conv2d':

                G, input_size = add_mp(G, input_size, param['name'], param_next['name'], param['kernel_size'], param['stride'], param['padding'], next_linear=False)

            if param_next['layer_type'] == 'Linear':

                G, input_size = add_mp(G, input_size, param['name'], param_next['name'], param['kernel_size'], param['stride'], param['padding'], next_linear=True)

        elif param['layer_type'] == 'Linear':
            # linear layer
            G = add_linear_linear(G, param['param'], param['name'], param_next['name'])

        else:
            raise ValueError('Layer type not implemented ')

    # add in last layer
    print('Layer: {}'.format(params[-1]['name']))
    G = add_linear_linear(G, params[-1]['param'], params[-1]['name'], 'Output')

    return G

def to_directed_networkx_activations(params, activations):
    '''Create networkx representation of activation graph of neural network. This
    function takes a list of parameter values and a list of activation values,
    both in the form of a list of numpy arrays (converted from pytorch tensor),
    one element in the list per layer.

    Args:
        params (list[numpy.array]): the weight (parameter) values of the network
            at each layer.
        activations (list[numpy.array]): the activation values of the network at
            each layer.

    Returns:
        networkx.DiGraph: The networkx representation of the activation network.
    '''
    # store all network info here
    G = nx.DiGraph()

    # get first input size
    input_size = activations[0].shape

    # assume last layer linear, loop over each layer and process
    for l in range(len(params)-1):

        # get parameter information for current layer
        param = params[l]
        # need to look ahead at next layer to get naming correct
        param_next = params[l+1]

        X = activations[l]

        print('Layer: {}'.format(param['name']))

        # check the layer type to decide how to process

        if param['layer_type'] == 'Conv2d':
            # convolutional layer

            if param_next['layer_type'] == 'Conv2d' or param_next['layer_type'] == 'MaxPool2d':
                # convolutional layer followed by a conv layer or maxpool layer

                # add edges and nodes of this layer to the networkx representation
                G, input_size = add_conv(G, input_size, param['param'], param['name'], param_next['name'], param['stride'], param['padding'], next_linear=False, X=X)

            elif param_next['layer_type'] == 'Linear':
                # convolutional layer followed by FC layer

                G, input_size = add_conv(G, input_size, param['param'], param['name'], param_next['name'], param['stride'], param['padding'], next_linear=True, X=X)

        elif param['layer_type'] == 'MaxPool2d':
            # max pooling layer

            if param_next['layer_type'] == 'Conv2d':
                # maxpool layer followed by convolutional layer

                G, input_size = add_mp_act(G, X, param['name'], param_next['name'], param['kernel_size'], param['stride'], param['padding'], next_linear=False)

            if param_next['layer_type'] == 'Linear':
                # maxpool layer followed by linear layer

                G, input_size = add_mp_act(G, X, param['name'], param_next['name'], param['kernel_size'], param['stride'], param['padding'], next_linear=True)

        elif param['layer_type'] == 'Linear':
            # linear layer. Assume next layer is also going to be linear
            G = add_linear_linear(G, param['param'], param['name'], param_next['name'], X=X)

        else:
            raise ValueError('Layer type not implemented ')

    # add in last layer and assume linear
    print('Layer: {}'.format(params[-1]['name']))
    G = add_linear_linear(G, params[-1]['param'], params[-1]['name'], 'Output', X=X)

    return G

def get_weights(model):
    params = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            pnum = param.data.cpu().numpy()
            params.append(pnum)
    return params

def append_params(param_info, params):
    for i in range(len(param_info)):
        p = param_info[i]
        if p['layer_type'] == 'Conv2d' or p['layer_type'] == 'Linear':
            p['param'] = params[i]
        else:
            p['param'] = None
    return param_info

def get_activations(model, data):
    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)
    def save_activation(name, mod, inp, out):
        activations[name].append(out.cpu())

    # Registering hooks for all layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    for name, m in model.named_modules():
        if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear or type(m) == torch.nn.MaxPool2d:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, name))

    # forward pass through the full dataset
    with torch.no_grad():
        out = model(data)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(outputs, 0).data.numpy() for name, outputs in activations.items()}

    return [v for k,v in activations.items()]

def parameter_graph(model, param_info, input_size):
    # get parameters from named parameters of model
    params = get_weights(model)

    # add `param` key to `param_info` list of dicts
    param_info = append_params(param_info, params)

    return to_directed_networkx(param_info, input_size)

def activation_graph(model, param_info, data):
    # get parameters from named parameters of model
    params = get_weights(model)
    # add `param` key to `param_info` list of dicts
    param_info = append_params(param_info, params)
    # get activations from network, given inputs in `data`
    activations = get_activations(model, data)

    return to_directed_networkx_activations(param_info, activations)
