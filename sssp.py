import numpy as np 
import networkx as nx 
import torch
from tqdm import tqdm 
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

def mp_estimate(ei: torch.Tensor, hops: int=10,  num_nodes: int=-1, verbose: bool=False) -> torch.Tensor:
    '''
    Use message passing to estimate the SSSP matrix
    and return D for the graph G. 
    Note: I think we can use MP to efficiently update the
    D matrix online? If the estimate is good enough, this is
    worth looking into 
    '''
    if num_nodes == -1:
        num_nodes = ei.max().item()+1
    
    x = torch.full((num_nodes, num_nodes), float('inf'), requires_grad=False)
    
    # Every node is reachable from itself 
    x[torch.arange(num_nodes), torch.arange(num_nodes)] = 0 
    x[ei[0], ei[1]] = 1

    mp = MessagePassing(aggr='min')
    for _ in tqdm(range(hops-1), disable=not verbose):
        # Probably a more efficient way to do this
        x = torch.min(x, mp.propagate(ei, size=None, x=x+1))

    return x 

# NX takes a rediculously long time
def ei_to_nx(ei):
    # Convert to nx graph 
    edge_list = [(ei[0][i].item(), ei[1][i].item()) for i in range(ei.size(1))]
    g = nx.from_edgelist(edge_list, create_using=nx.DiGraph)
    return g 

def nx_djikstra(g: nx.DiGraph): 
    # Returns sparse representation of the form 
    # [ (nid, {neigbhor: distance}) ]
    return [i for i in nx.shortest_path_length(g)]


def eigen_product(g: Data, pow=10) -> torch.Tensor:
    '''
    Get path info by taking the adj mat to 
    an arbitrarilly high power. Not actually SSSP, instead the number of walks
    from (i,j) of len pow (as approaches inf stabilizes)
    '''
    adj = to_dense_adj(g.edge_index)[0].numpy()

    # Add self-loops and norm so eigs are real and result is \in (0,1)
    adj[range(adj.shape[0]), range(adj.shape[0])] = 1 
    adj = adj / adj.sum(axis=1)[:,None]

    val, vec = np.linalg.eig(adj)
    val = np.diag(np.power(val, pow))

    return torch.from_numpy(
        vec @ val @ np.linalg.inv(vec)
    )

if __name__ == '__main__':
    ei = torch.tensor([[0,0,1,1,1,2,2,3,3,4,4,4,5,5,6,7,7,8,8,9,9,10,10],[1,2,0,3,7,0,5,1,5,3,5,6,2,4,4,1,4,9,10,8,10,8,9]])
    g = Data(edge_index=ei, num_nodes=8)