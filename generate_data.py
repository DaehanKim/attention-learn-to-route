import argparse
import os
from posixpath import relpath
import numpy as np
from utils.data_utils import check_extension, save_dataset
from tqdm import tqdm
import pickle


def generate_dataset_based_on_pkl(pkl_path, dataset_size, question_num, student_num, seed=1234):
    with open(pkl_path, 'rb') as f: d = pickle.load(f) 
    np.random.seed(seed)
    arr = np.array(d)
    data = []
    pert = arr.std() * 0.05
    _, n_question, n_student = arr.shape
    pert_arr = np.random.randn(question_num, n_student)
    for i in tqdm(range(dataset_size)):
        if question_num < n_question:
            idx = np.random.randint(0, n_question-question_num)
            ret = arr[0,idx:idx+question_num] + pert_arr*pert*(0.5-i/dataset_size) # make perturbed data
            
        else:
            ret = [arr[0].copy() + np.random.randn(n_question, n_student)*pert for _ in range(int(question_num/n_question))]
            rest = question_num % n_question
            ret.append(arr[0][:rest] + np.random.randn(rest, n_student)*pert)
            ret = np.concatenate(ret, axis=0)
        data.append(ret)        
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--dataset_size", type=int, default=2000, help="Size of the dataset")
    parser.add_argument("--question_num", type=int, default=500, help="Size of the dataset")
    parser.add_argument("--student_num", type=int, default=2000, help="Size of the dataset")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()


    problem = 'dg'
    datadir = os.path.join(opts.data_dir, problem)
    os.makedirs(datadir, exist_ok=True)

    filename = os.path.join(datadir, f"q{opts.question_num}_s{opts.student_num}_n{opts.dataset_size}_seed{opts.seed}_basedon")

    assert opts.f or not os.path.isfile(check_extension(filename)), \
        "File already exists! Try running with -f option to overwrite."

    snapshot_dir = '/workspace/attention-learn-to-route/data/dg'
    for path in [os.path.join(snapshot_dir,p) for p in os.listdir(snapshot_dir) if 'train' in p]:
        symbol = os.path.split(path)[-1].split('.')[0].replace("_","").replace("train", "")
        data = generate_dataset_based_on_pkl(path, opts.dataset_size, opts.question_num, opts.student_num, opts.seed)
        assert len(np.array(data).shape) == 3
        save_dataset(data, filename + f'_{symbol}.pkl')



    
