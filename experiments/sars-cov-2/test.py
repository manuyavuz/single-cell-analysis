from geomloss import SamplesLoss
import argparse
import torch
from joblib import Parallel, parallel_backend, delayed
from tqdm.auto import tqdm

device = torch.device('cpu')
sinkhorn_geomloss = SamplesLoss(loss="sinkhorn", scaling=0.9, p=1, debias=True).to(device)

def run(i):
    sinkhorn_geomloss(torch.randn(100,5), torch.randn(200,5))
    
def test_sequential():
    for i in tqdm(range(100)):
        sinkhorn_geomloss(torch.randn(100,5), torch.randn(200,5))

def test_threading(args):
    with parallel_backend('threading'):
        Parallel(verbose=1, n_jobs=args.n_jobs)(delayed(run)(i) for i in tqdm(range(100)));
    
def test_loky(args):
    with parallel_backend('loky'):
        Parallel(verbose=1, n_jobs=args.n_jobs)(delayed(run)(i) for i in tqdm(range(100)));

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--type')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--n_torch_threads', type=int)
    args = parser.parse_args()
    if args.n_torch_threads is not None:
        torch.set_num_threads(args.n_torch_threads)
    if args.type == 'seq':
        test_sequential()
    elif args.type == 'threading':
        test_threading(args)
    elif args.type == 'loky':
        test_loky(args)

if __name__ == "__main__":
    main()