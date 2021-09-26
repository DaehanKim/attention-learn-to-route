import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateDG(NamedTuple):
    # Fixed input
    loc: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    diag_size : torch.Tensor # store # problems for diagnosis
    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    # lengths: torch.Tensor
    std : torch.Tensor
    rmse : torch.Tensor
    fitness : torch.Tensor
    cur_ques: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    visit_idx : torch.Tensor
    total_mean : torch.Tensor # (batch_dim, n_student)
    lamb : torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            std = self.std[key],
            rmse = self.rmse[key],
            fitness = self.fitness[key],
            cur_ques=self.cur_ques[key] if self.cur_ques is not None else None,
            visit_idx = self.visit_idx[key],
            total_mean = self.total_mean[key],
            lamb = self.lamb
        )

    @staticmethod
    def initialize(loc, diag_size=10, visited_dtype=torch.uint8, lamb = torch.tensor([0.3547])):

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        total_mean = loc.mean(dim=1)
        return StateDG(
            loc=loc,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            std = torch.zeros(batch_size, 1, device=loc.device),
            rmse = torch.zeros(batch_size, 1, device=loc.device),
            fitness = torch.zeros(batch_size, 1, device=loc.device),
            cur_ques=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            diag_size= torch.empty(1).fill_(diag_size).to(loc.device),
            visit_idx = [],
            total_mean = total_mean,
            lamb = lamb.to(loc.device)
        )

    def get_final_cost(self):

        assert self.all_finished()

        return -self.fitness

    def update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step
        self.visit_idx.append(prev_a.clone()) # store selected node for each snapshot
        indices = torch.cat(self.visit_idx, dim=1).long() # (batch_dim, n_step)
        collected_snapshot = []
        for idx, b_idx in enumerate(indices):
            collected_snapshot.append(self.loc[idx][b_idx])
        collected_snapshot = torch.stack(collected_snapshot, dim=0) # (batch_dim, n_step, n_student)
        cur_ques = self.loc[self.ids, prev_a]
        rmse = self.rmse 
        std = self.std
        fitness = self.fitness
        if self.cur_ques is not None:  
            rmse =  ((self.total_mean - collected_snapshot.mean(dim=1)).pow(2)).mean(1, keepdim=True).pow(0.5)  # (batch_dim, 1)
            std =   collected_snapshot.mean(1).std(1, keepdim=True) # (batch_dim, 1)
            fitness = -rmse + std * self.lamb

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_,
                             rmse=rmse, std = std , fitness = fitness, cur_ques=cur_ques, i=self.i + 1)

    def all_finished(self):
        # predefined diagnosis size
        return self.i.item() >= self.diag_size.item()

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions
