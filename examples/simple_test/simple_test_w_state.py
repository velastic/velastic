from velastic.manager import IterManager
from velastic.state_dict import DCPState
import time
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_training(rank, world_size):
    setup(rank, world_size)
    
    model = SimpleModel().to(rank)
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.01)]
    
    iter_mngr = IterManager(velastic_dir="./examples/.velastic")
    state = DCPState(model=model, optimizer=optimizers[0], framework="ddp", default_ckpt_path="./examples/.velastic/ckpt")
    
    # train
    while True:
        iter_mngr.iter()
        print(f"Rank {rank}: Iteration {iter_mngr.iter_count}")
        time.sleep(1)
        if iter_mngr.should_save_ckpt():
            state.save()
        if float(os.environ.get("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP", time.time() + 10)) < time.time():
            break
    
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running training example on {world_size} devices.")
    mp.spawn(
        run_training,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )