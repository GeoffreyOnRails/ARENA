import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import tests
from torch import Tensor

WORLD_SIZE = torch.cuda.device_count()

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"


def send_receive(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Send tensor to rank 1
        sending_tensor = torch.zeros(1)
        print(f"{rank=}, sending {sending_tensor=}")
        dist.send(tensor=sending_tensor, dst=1)
    elif rank == 1:
        # Receive tensor from rank 0
        received_tensor = torch.ones(1)
        print(f"{rank=}, creating {received_tensor=}")
        dist.recv(
            received_tensor, src=0
        )  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, received {received_tensor=}")

    dist.destroy_process_group()

def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank 0 to all other ranks.
    """
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    for send_rank in range(1,world_size):
        if rank == src:
        # Send tensor to rank 1
            dist.send(tensor=tensor, dst=send_rank)
        else:
            # Receive tensor from rank 0
            received_tensor = torch.ones(1)
            print(f"{rank=}, creating {received_tensor=}")
            dist.recv(
                received_tensor, src=src
            )  # this line overwrites the tensor's data with our `sending_tensor`
            print(f"{rank=}, received {received_tensor=}")


if __name__ == "__main__":
    world_size = 3  # simulate 2 processes
    tests.test_broadcast(broadcast, world_size)
    # print("coucou")
    # mp.spawn(
    #     send_receive,
    #     args=(world_size,),
    #     nprocs=world_size,
    #     join=True,
    # )