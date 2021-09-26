import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
from dg_data import Simulated_k
from tqdm import tqdm
import random


def generate_dg_data(dataset_size, question_num, student_num, seed=1234):
    train, val, test = [], [], []
    for idx in tqdm(range(dataset_size), total=dataset_size, desc="Generating Snapshots"):
        _tr, _val, _te = Simulated_k(n=student_num*3, q=question_num, seed=seed+idx).get_data()
        train.append(_tr.T.tolist())
        val.append(_val.T.tolist())
        test.append(_te.T.tolist())
    return train, val, test

def generate_dg_train_data(dataset_size, question_num, student_num, seed=None):
    seed = random.randint(1235,10000) if seed is None else seed
    data = []
    for idx in tqdm(range(dataset_size), total=dataset_size, desc="Generating Snapshots"):
        _tr = Simulated_k(n=student_num, q=question_num, seed=seed+idx).snapshot.T
        data.append(_tr.tolist())
    return data

def generate_random_matrix_data(dataset_size, question_num, student_num, seed=None):
    seed = random.randint(0,10000) if seed is None else seed 
    return np.random.uniform(0.4,1,(dataset_size, question_num, student_num)).tolist()

def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Size of the dataset")
    parser.add_argument("--question_num", type=int, default=50, help="Size of the dataset")
    parser.add_argument("--student_num", type=int, default=100, help="Size of the dataset")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()


    problem = 'dg'
    datadir = os.path.join(opts.data_dir, problem)
    os.makedirs(datadir, exist_ok=True)

    if opts.filename is None:
        filename = os.path.join(datadir, f"dg_q{opts.question_num}_seed{opts.seed}_{opts.name}")
    else:
        filename = check_extension(opts.filename)

    assert opts.f or not os.path.isfile(check_extension(filename)), \
        "File already exists! Try running with -f option to overwrite."

    dataset_train, dataset_valid, dataset_test = generate_dg_data(opts.dataset_size, opts.question_num, opts.student_num, opts.seed)

    print(dataset_train[0])

    save_dataset(dataset_train, filename + '_train.pkl')
    save_dataset(dataset_valid, filename + '_valid.pkl')
    save_dataset(dataset_test, filename + '_test.pkl')
