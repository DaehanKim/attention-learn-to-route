from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.dg.state_diagnosis import StateDG
from utils.beam_search import beam_search
from .generate_data import generate_random_matrix_data, generate_dg_train_data

class DG(object):
    # shortcut for Diagnosis Generation

    NAME = 'dg'

    @staticmethod
    def get_costs(dataset, pi, lamb=0.06):
        # Check that tours are valid, i.e. contain 0 to n -1
        

        # Gather dataset in order of tour
        collected_snapshot = []
        for idx, b_idx in enumerate(pi):
            collected_snapshot.append(dataset[idx][b_idx])
        collected_snapshot = torch.stack(collected_snapshot, dim=0) # (batch_dim, n_step, n_student)
        
        rmse =  ((dataset.mean(1) - collected_snapshot.mean(dim=1)).pow(2)).mean(1, keepdim=True).pow(0.5)  # (batch_dim, 1)
        std =   collected_snapshot.mean(1).std(1, keepdim=True) # (batch_dim, 1)
        fitness = -rmse + std * lamb
        return rmse, std, -fitness, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return DGDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateDG.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = DG.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class DGDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=2000, offset=0):
        super(DGDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # raise (NotImplementedError, "Generating data on-the-fly is not allowed in DG task.")
            self.data = [torch.FloatTensor(item) for item in generate_random_matrix_data(num_samples, 50, 100, seed = None)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
