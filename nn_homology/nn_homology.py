from ripser import ripser
from nn_homology.nn_graph import parameter_graph, activation_graph

def parameter_homology(model, param_info, input_size, max_dim=1, do_cocycles=False):
    '''Uses scikit-tda's ripser implementation to compute parameter graph
    persistent homology of the model network. First computes the parameter
    graph then computes homology.

    Args:
        model (pytorch model): the model from which to form the parameter graph.
        param_info (list[dict]): a list of dictionaries describing the model's
            architecture.
        input_size (tuple): the size of the input at the first layer in form
            (batch, channels, height, width).
        max_dim (int, optional): the maximum dimension to compute homology up to.
            Default is 1.
        do_cocycles (bool, optional): whether to return cocycles of persistent
            (co)homology calculation. Default is False.

    Returns:
        dict: the scikit-tda dictionary output. See: https://ripser.scikit-tda.org/reference/stubs/ripser.ripser.html#ripser.ripser
    '''
    print('Computing parameter graph ...')
    G = parameter_graph(model, param_info, input_size)
    print('Computing persistent homology ... ')
    return ripser(nx.to_scipy_sparse_matrix(G), distance_matrix=True, maxdim=max_dim, do_cocycles=do_cocycles)

def activation_homology(model, param_info, data, max_dim=1, do_cocycles=False):
    '''Uses scikit-tda's ripser implementation to compute activation graph
    persistent homology of the model network. First computes the activation
    graph then computes homology.

    Args:
        model (pytorch model): the model from which to form the parameter graph.
        param_info (list[dict]): a list of dictionaries describing the model's
            architecture.
        data (torch.Tensor): data (torch.Tensor): an input to the model.
        max_dim (int, optional): the maximum dimension to compute homology up to.
            Default is 1.
        do_cocycles (bool, optional): whether to return cocycles of persistent
            (co)homology calculation. Default is False.

    Returns:
        dict: the scikit-tda dictionary output. See: https://ripser.scikit-tda.org/reference/stubs/ripser.ripser.html#ripser.ripser
    '''
    print('Computing activation graph ...')
    G = activation_graph(model, param_info, data)
    print('Computing persistent homology ...')
    return ripser(nx.to_scipy_sparse_matrix(G), distance_matrix=True, maxdim=max_dim, do_cocycles=do_cocycles)
