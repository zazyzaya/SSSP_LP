import glob 
import gzip 
import time 

import torch 
from tqdm import tqdm 

from sssp import mp_estimate

IN_F = '/mnt/raid1_ssd_4tb/datasets/LANL15/torch_data/'
OUT_F = 'data/'
N_FILES = 1392
N_NODES = 23156

@torch.no_grad()
def remove_dupe_edges(ei, return_index=False):
    # Produces unique "hash" for each src,dst pair
    encoded = ei[0] + N_NODES*ei[1]
    uq_edges_enc = encoded.unique(return_inverse=return_index)

    idx = None 
    if return_index:
        uq_edges_enc, idx = uq_edges_enc

    # Recover original values from unique tensor and return
    new_ei = torch.stack([
        uq_edges_enc % N_NODES, 
        uq_edges_enc // N_NODES
    ])

    if return_index:
        return new_ei, idx 
    return new_ei 

def get_data_split():
    '''
    Comparing to Ben's paper. 
    Find 40 fully benign days, 18 mal days for tr/te split 
    '''
    mal_days = set()
    red = gzip.open(IN_F + '../redteam.txt.gz', 'rt')

    line = red.readline()
    while line: 
        ts = int(line.split(',')[0])
        mal_days.add(ts // (60*60*24))
        line = red.readline()

    ben_days = list(set(range(58)) - mal_days)
    mal_days = list(mal_days)
    
    print(mal_days) 
    return ben_days, mal_days 

def build_benign_graph(): 
    benign, _ = get_data_split()
    b_hours = sum([ [r+(i*24) for r in range(24)] for i in benign ], [])

    ei = torch.load(IN_F + f'{b_hours.pop()}.pt').edge_index
    ei = remove_dupe_edges(ei)

    for hour in tqdm(b_hours): 
        new_ei = torch.load(IN_F + f'{hour}.pt').edge_index
        ei = remove_dupe_edges(torch.cat([ei, new_ei], dim=1))
    
    print('Saving')
    torch.save(ei, OUT_F + 'benign.pt')
    return ei 

def test_one_hour(benign_assp, hour):
    # Remove infinities
    benign_assp += 1
    benign_assp[benign_assp == float('inf')] = 0 

    mal = torch.load(IN_F + f'{hour}.pt')
    mal_ei = mal.edge_index 
    y = mal.y 
    del mal  

    uq_mal, idx = remove_dupe_edges(mal_ei, return_index=True)
    dists = (benign_assp[uq_mal[0]] - benign_assp[uq_mal[1]]) ** 2
    dists = dists.sum(dim=1).float() ** (1/2)

    dists = dists[idx]
    return dists, y 