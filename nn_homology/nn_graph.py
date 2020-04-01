import collections
from functools import partial

import numpy as np
import networkx as nx
import scipy.sparse
from scipy.spatial.distance import squareform
import torch

def inverse_abs(x):
    return np.abs(1/x)

def inverse_abs_zero(x):
    return np.abs(1/(1+x))

def format_func(name, channel, i):
    return '{}_{}_{}'.format(name, channel, i)

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

def add_conv(G, input_size, p, name_this, name_next, stride, padding, next_linear=False,
            X=None, I=0, format_func=format_func, weight_func=inverse_abs_zero,
            ignore_zeros=False):
    '''Adds a convolutional layer to graph and returns updated graph. If X is
    given, compute the activation graph, otherwise compute the parameter graph.

    Args:
        G (networkx.Graph): the networkx representation of the graph to which to
            add the linear layer.
        input_size (tuple): the input size to the layer in form
            (batch, channels, height, width).
        p (numpy.array): a numpy representation of the parameters of the layer
            of dimensionality [|l+1|,|l|] where |l| represents the number of
            nodes in layer l and |l+1| represents the number of nodes in the
            following layer.
        name_this (str): the name of layer l.
        name_next (str): the name of layer l+1.
        stride (int): the stride of the convolutional operation.
        padding (int): the padding of the convolutional operation.
        next_linear (bool, optional): boolean for whether the next layer is linear.
        X (numpy.array, optional): the activation values at each node in layer l.
        I (int, optional): the running index value.
        format_func (callable, optional): node name format function. This
            function should take three arguments (str, float, float) and return
            a string.
        weight_func (callable, optional): a function to apply to the weight values.
            Should take a single float-like argument and return a float-like argument.

    Returns:
        (networkx.Graph, tuple, int): a tuple representing the updated networkx graph,
            the new input size, and the running parameter index.
    '''
    input_channels = p.shape[1]
    X = np.ones(input_size) if X is None else X

    X_names = np.arange(X.shape[2]*X.shape[3]).reshape((1,1,X.shape[2],X.shape[3]))
    tx = X
    filt_size = p.shape[2]*p.shape[3]
    # convert to matrix information
    mat, X_col, W_col, xnames = conv_layer_as_matrix(tx,X_names,p,stride,padding)
    for f in range(W_col.shape[0]):
        for row in range(X_col.shape[0]):
            c = row//filt_size
            wix = I + f*W_col.shape[1] + row
            for col in range(X_col.shape[1]):
                v = W_col[f,row] * X_col[row,col]
                if not ignore_zeros or v != 0:
                    if next_linear:
                        # next layer is linear
                        G.add_edge(format_func(name_this,c,xnames[row%filt_size,col]),format_func(name_next,0,int((X_col.shape[1]*c) + (f*X_col.shape[1]) + col)), weight=weight_func(v), idx=wix)
                    else:
                        # next layer is conv
                        G.add_edge(format_func(name_this,c,xnames[row%filt_size,col]),format_func(name_next,f,col), weight=weight_func(v), idx=wix)
    input_size = mat.shape
    return G, input_size, wix+1

# under construction
def add_mp(G, input_size, name_this, name_next, kernel_size, stride, padding, next_linear=False, format_func=format_func):
    '''adds max pooling layer to graph and returns updated graph'''
    p = np.ones((input_size[0],input_size[1],kernel_size[0],kernel_size[1]))
    input_channels = p.shape[1]
    # next layer also conv
    X = np.ones(input_size)
    X_names = np.arange(X.shape[2]*X.shape[3]).reshape((1,1,X.shape[2],X.shape[3]))
    tx = X
    filt_size = kernel_size[0]*kernel_size[1]
    # convert to matrix information
    mat, X_col, W_col, xnames = conv_layer_as_matrix(tx,X_names,p,stride,padding)
    for f in range(W_col.shape[0]):
        for row in range(X_col.shape[0]):
            c = row//filt_size
            for col in range(X_col.shape[1]):
                node_name = format_func(name_this,c,xnames[row%filt_size,col])
                ews = sorted(G.in_edges(node_name, data=True), key=lambda x: x[2]['weight'])
                if len(ews) > 0:
                    # grab smallest value as this should correspond to largest
                    # weight before `weight_func` is applied.
                    v = ews[0][2]['weight']
                    wix = ews[0][2]['idx']
                    if next_linear:
                        G.add_edge(format_func(name_this,c,xnames[row%filt_size,col]),format_func(name_next,0,int((X_col.shape[1]*c) + (f*X_col.shape[1]) + col)), weight=v, idx=wix)
                    else:
                        G.add_edge(format_func(name_this,c,xnames[row%filt_size,col]),format_func(name_next,f,col), weight=v, idx=wix)
    input_size = [mat.shape[0], input_channels, mat.shape[2], mat.shape[3]]
    return G, input_size

# under construction
def add_mp_act(G, X, name_this, name_next, kernel_size, stride, padding, weight_func=inverse_abs, next_linear=False):
    '''adds max pooling layer to graph and returns updated activation graph'''
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

def add_linear_linear(G, p, name_this, name_next, X=None, I=0, format_func=format_func,
                    weight_func=inverse_abs_zero, ignore_zeros=False):
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
        X (numpy.array, optional): the activation values at each node in layer l.
        I (int, optional): the running index value.
        format_func (callable, optional): node name format function. This
            function should take three arguments (str, float, float) and return
            a string.
        weight_func (callable, optional): a function to apply to the weight values.
            Should take a single float-like argument and return a float-like argument.

    Returns:
        (networkx.Graph, int): the updated networkx graph and the running
            parameter index.
    '''
    p = p if X is None else p*X
    for row in range(p.shape[1]):
        for col in range(p.shape[0]):
            wix = I + col*p.shape[1] + row
            v = p[col,row]
            if not ignore_zeros or v != 0:
                G.add_edge(format_func(name_this,0,row),format_func(name_next,0,col), weight=weight_func(v), idx=wix)
    return G, wix+1

def to_directed_networkx(params, input_size, format_func=format_func,
                        weight_func=inverse_abs_zero, ignore_zeros=False):
    '''Create networkx representation of parameter graph of neural network. This
    function takes a list of parameter values and a list of activation values,
    both in the form of a list of numpy arrays (converted from pytorch tensor),
    one element in the list per layer.

    Args:
        params (list[numpy.array]): the weight (parameter) values of the network
            at each layer.
        input_size (tuple): the size of the first layer's input in form
            (batch, channels, height, width).
        format_func (callable, optional): node name format function. This
            function should take three arguments (str, float, float) and return
            a string.
        weight_func (callable, optional): a function to apply to the weight values.
            Should take a single float-like argument and return a float-like argument.

    Returns:
        networkx.DiGraph: The networkx representation of the parameter network.
    '''
    # store all network info here
    G = nx.DiGraph()

    # index into flattened parameters
    I = 1

    # assume last layer linear, loop over each layer and process
    for l in range(len(params)-1):

        # get parameter information for current layer
        param = params[l]
        # need to look ahead at next layer to get naming correct
        param_next = params[l+1]
        # shortcuts are not actually layers, so skip the name
        if param_next['layer_type'] == 'Shortcut':
            param_next = params[l+2]

        print('Layer: {}'.format(param['name']))

        # check the layer type to decide how to process
        if param['layer_type'] == 'Conv2d':

            if param_next['layer_type'] == 'Conv2d' or param_next['layer_type'] == 'MaxPool2d':
                # add edges and nodes of this layer to the networkx representation
                G, input_size, I = add_conv(G, input_size, param['param'], param['name'],
                                    param_next['name'], param['stride'],
                                    param['padding'], next_linear=False, I=I,
                                    format_func=format_func, weight_func=weight_func,
                                    ignore_zeros=ignore_zeros)

            elif param_next['layer_type'] == 'Linear':

                G, input_size, I = add_conv(G, input_size, param['param'], param['name'],
                                    param_next['name'], param['stride'],
                                    param['padding'], next_linear=True, I=I,
                                    format_func=format_func, weight_func=weight_func,
                                    ignore_zeros=ignore_zeros)

        elif param['layer_type'] == 'MaxPool2d':

            if param_next['layer_type'] == 'Conv2d':

                G, input_size = add_mp(G, input_size, param['name'], param_next['name'],
                                    param['kernel_size'], param['stride'],
                                    param['padding'], next_linear=False,
                                    format_func=format_func)

            if param_next['layer_type'] == 'Linear':

                G, input_size = add_mp(G, input_size, param['name'], param_next['name'],
                                    param['kernel_size'], param['stride'],
                                    param['padding'], next_linear=True,
                                    format_func=format_func)

        elif param['layer_type'] == 'Linear':
            # linear layer
            G, I = add_linear_linear(G, param['param'], param['name'], param_next['name'],
                                    I=I, format_func=format_func, weight_func=weight_func,
                                    ignore_zeros=ignore_zeros)

        elif param['layer_type'] == 'Shortcut':

            # get the layers the shortcut connects
            from_name = params[param['connects'][0]]
            to_name = params[param['connects'][1]]
            G, input_size, I = add_conv(G, input_size, param['param'], from_name,
                                to_name, param['stride'],
                                param['padding'], next_linear=False, I=I,
                                format_func=format_func, weight_func=weight_func,
                                ignore_zeros=ignore_zeros)

        else:
            raise ValueError('Layer type not implemented ')

    # add in last layer
    print('Layer: {}'.format(params[-1]['name']))
    G, I = add_linear_linear(G, params[-1]['param'], params[-1]['name'], 'Output',
                            I=I, format_func=format_func, weight_func=weight_func,
                            ignore_zeros=ignore_zeros)
    return G

def to_directed_networkx_activations(params, activations, format_func=format_func,
                        weight_func=inverse_abs_zero, ignore_zeros=False):
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
                G, input_size = add_conv(G, input_size, param['param'], param['name'],
                                    param_next['name'], param['stride'],
                                    param['padding'], next_linear=False, X=X,
                                    ignore_zeros=ignore_zeros)

            elif param_next['layer_type'] == 'Linear':
                # convolutional layer followed by FC layer

                G, input_size = add_conv(G, input_size, param['param'], param['name'],
                                    param_next['name'], param['stride'],
                                    param['padding'], next_linear=True, X=X,
                                    ignore_zeros=ignore_zeros)

        elif param['layer_type'] == 'MaxPool2d':
            # max pooling layer

            if param_next['layer_type'] == 'Conv2d':
                # maxpool layer followed by convolutional layer

                G, input_size = add_mp_act(G, X, param['name'], param_next['name'],
                                    param['kernel_size'], param['stride'],
                                    param['padding'], next_linear=False)

            if param_next['layer_type'] == 'Linear':
                # maxpool layer followed by linear layer

                G, input_size = add_mp_act(G, X, param['name'], param_next['name'],
                                    param['kernel_size'], param['stride'],
                                    param['padding'], next_linear=True)

        elif param['layer_type'] == 'Linear':
            # linear layer. Assume next layer is also going to be linear
            G = add_linear_linear(G, param['param'], param['name'], param_next['name'], X=X)

        else:
            raise ValueError('Layer type not implemented ')

    # add in last layer and assume linear
    print('Layer: {}'.format(params[-1]['name']))
    G = add_linear_linear(G, params[-1]['param'], params[-1]['name'], 'Output',
                        X=X, ignore_zeros=ignore_zeros)

    return G

def get_weights(model, tensors=False):
    '''Helper function to retrieve named weights from pytorch model.'''
    params = []
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name:
            if tensors:
                pnum = param.data
            else:
                pnum = param.data.cpu().numpy()
            params.append(pnum)
    return params

def append_params(param_info, params):
    '''Helper function to append pytorch parameters to parameter information.'''
    j = 0
    for i in range(len(param_info)):
        p = param_info[i]
        if p['layer_type'] == 'Conv2d' or p['layer_type'] == 'Linear' or p['layer_type'] == 'Shortcut':
            # if ndim == 1, probably batch norm, pass
            if params[j].ndim > 1:
                p['param'] = params[j]
            else:
                j += 1
                p['param'] = params[j]
            j += 1
        else:
            p['param'] = None

    return param_info

def get_activations(model, data):
    '''Retrieves and returns the activations at each layer given an input `data`.

    Args:
        model (pytorch model): the model from which to retrieve activations.
        data (torch.Tensor): the input whose activation values we are interested
            in.

    Returns:
        list[torch.Tensor]: a list of torch tensors representing the activations
            at each layer.
    '''
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
    '''Returns a networkx DiGraph representation of the model's parameter graph.

    Args:
        model (pytorch model): the model from which to form the parameter graph.
        param_info (list[dict]): a list of dictionaries describing the model's
            architecture.
        input_size (tuple): the size of the input at the first layer in form
            (batch, channels, height, width).

    Returns:
        networkx.DiGraph: the parameter graph representation of the model.
    '''
    # get parameters from named parameters of model
    params = get_weights(model)

    # add `param` key to `param_info` list of dicts
    param_info = append_params(param_info, params)

    return to_directed_networkx(param_info, input_size)

def activation_graph(model, param_info, data):
    '''Returns a networkx DiGraph representation of the model's activation graph
    for a given input `data`.

    Args:
        model (pytorch model): the model from which to form the parameter graph.
        param_info (list[dict]): a list of dictionaries describing the model's
            architecture.
        data (torch.Tensor): an input to the model.

    Returns:
        networkx.DiGraph: the activation graph representation of the model.
    '''
    # get parameters from named parameters of model
    params = get_weights(model)
    # add `param` key to `param_info` list of dicts
    param_info = append_params(param_info, params)
    # get activations from network, given inputs in `data`
    activations = get_activations(model, data)

    return to_directed_networkx_activations(param_info, activations)

def flatten_params(param_info):
    '''Flattens parameter vectors so as to match the flattening that happens
    when computing the graphical representation of the network.

    Args:
        param_info (list[dict]): a list of dictionaries describing the model's
            architecture. Each dict should have a `param` key.

    Returns:
        np.array: the parameters of the network in a flattened, 1D array.
    '''
    param_vecs = []
    for param in param_info:
        if param['layer_type'] == 'Conv2d':
            p = param['param']
            param_vecs.append(p.reshape(p.shape[0],-1).flatten())
        if param['layer_type'] == 'Linear':
            p = param['param']
            param_vecs.append(p.flatten())

    # make the first element zero (could be anything given we're filtering below)
    param_vecs = [np.array([0.])] + param_vecs
    param_vec = np.concatenate(param_vecs)

    return param_vec

class NNGraph(object):

    def __init__(self, weight_func=inverse_abs_zero, format_func=format_func,
                undirected=True):
        '''NNGraph class for computing parameter graph of network.

        Args:
            weight_func (callable, optional): function to be applied to weights
                before being added to graph. This function should act element-wise
                on numpy arrays.
            format_func (callable, optional): node name format function. This
                function should take three values and return a string.
            undirected (bool, optional): whether to construct the undirected
                version of the graph.
        '''
        self.G = nx.DiGraph()
        self.adj = None
        self.graph_idx_vec = None
        self.adj_vec = None
        self.current_param_info = None
        self.input_size = None
        self.weight_func = weight_func
        self.format_func = format_func
        self.undirected = undirected

    def update_indices(self):
        self.graph_idx_vec = np.array(nx.to_numpy_matrix(self.G, weight='idx', dtype='int'))[np.tril_indices(len(self.G.nodes()),-1)]
        self.adj_vec = np.array(nx.to_numpy_matrix(self.G, weight='weight'))[np.tril_indices(len(self.G.nodes()),-1)]

    def parameter_graph(self, model, param_info, input_size, ignore_zeros=False, update_indices=False):
        '''Returns a networkx DiGraph representation of the model's parameter graph.
        Also instantiates the class's internal representations of the network.

        Args:
            model (pytorch model): the model from which to form the parameter graph.
            param_info (list[dict]): a list of dictionaries describing the model's
                architecture.
            input_size (tuple): the size of the input at the first layer in form
                (batch, channels, height, width).
            ignore_zeros (bool, optional): whether to ignore edges with weight
                zero.

        Returns:
            networkx.DiGraph: the parameter graph representation of the model.
        '''
        self.input_size = input_size
        # get parameters from named parameters of model
        params = get_weights(model)
        # add `param` key to `param_info` list of dicts
        param_info = append_params(param_info, params)

        self.current_param_info = param_info
        G  = to_directed_networkx(self.current_param_info, self.input_size, format_func=self.format_func, weight_func=self.weight_func, ignore_zeros=ignore_zeros)
        if self.undirected:
            self.G = G.to_undirected()
        if update_indices:
            self.update_indices()
        return G

    def symmetrize(self):
        self.G = self.G.to_undirected()
        self.update_indices()

    def get_adjacency(self):
        '''returns adjacency matrix from its flattened form.'''
        num_nodes = len(self.G.nodes())
        i,j = np.tril_indices(num_nodes,-1)
        R = np.zeros((num_nodes,num_nodes))
        R[i,j] = self.adj_vec
        R[j,i] = self.adj_vec
        return R
        # return squareform(self.adj_vec)

    def get_idx_adjacency(self):
        '''returns adjacency matrix from its flattened form.'''
        sq = int(np.sqrt(self.graph_idx_vec.size))
        return np.reshape(self.graph_idx_vec, (sq, sq))

    def update_adjacency(self, model):
        '''Updates the parameter adjacency matrix given a model's parameters.
        This function assumes the input model is the same architecture as that
        the NNGraph instantiation is based on.

        Args:
            model (pytorch model): the model from which to extract parameters and
                update the adjacency matrix.
        '''
        params = get_weights(model)
        # add `param` key to `param_info` list of dicts
        param_info = append_params(self.current_param_info, params)
        self.current_param_info = param_info

        param_vec = flatten_params(param_info)

        self.param_vec = param_vec
        self.adj_vec[self.graph_idx_vec > 0] = self.weight_func(np.squeeze(np.take(param_vec, self.graph_idx_vec[self.graph_idx_vec > 0])))

    def update_graph(self):
        node_names = list(self.G.nodes())
        name_dict = { i : node_names[i] for i in range(len(node_names)) }
        adj = self.get_adjacency()
        cu = nx.Graph() if self.undirected else nx.DiGraph()
        G = nx.relabel_nodes(nx.from_numpy_matrix(adj, create_using=cu), name_dict)
        self.G = G
        return G
