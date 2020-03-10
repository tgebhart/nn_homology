from ripser import ripser
from nn_homology.nn_graph import parameter_graph, activation_graph

def parameter_homology(model, param_info, input_size, max_dim=1):
    print('Computing parameter graph ...')
    G = parameter_graph(model, param_info, input_size)
    print('Computing homology ... ')
    return ripser(nx.to_scipy_sparse_matrix(G), distance_matrix=True, maxdim=max_dim)

def activation_homology(model, param_info, data, max_dim=1):
    print('Computing activation graph ...')
    G = activation_graph(model, param_info, data)
    print('Computing homology ...')
    return ripser(nx.to_scipy_sparse_matrix(G), distance_matrix=True, maxdim=max_dim)
