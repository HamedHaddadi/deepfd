from os import path 

# method to examine computational graphs 
def measure_graph_size(func, *inputs):
    """
    computes length of computational graphs
        to avoid blow ups in eager executions 
    """
    g = func.get_concrete_function(*inputs).graph 
    return func.__name__, len(g.as_graph_def().node)