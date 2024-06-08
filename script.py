import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torch.nn as nn
import torch.optim as optim

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.fc(x)

# Training function
def train(rref):
    model = rref.local_value()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(100):  # Replace with your desired number of epochs
        inputs = torch.randn(10)
        labels = torch.randn(10)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    return model.state_dict()

def run_worker(rank, world_size):
    # Initialize RPC
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    
    if rank == 0:
        # Master node: create the model and distribute work
        model = MyModel()
        model_rref = RRef(model)
        futs = []
        for worker in range(1, world_size):
            futs.append(rpc.rpc_async(f"worker{worker}", train, args=(model_rref,)))
        # Collect results
        for fut in futs:
            state_dict = fut.wait()
            model.load_state_dict(state_dict)
    else:
        # Worker nodes: do nothing, training is handled by the train function
        pass
    
    # Shutdown RPC
    rpc.shutdown()

if __name__ == "__main__":
    import os
    import torch.multiprocessing as mp
    world_size = 4  # Total number of nodes

    mp.spawn(run_worker,
             args=(world_size,),
             nprocs=world_size,
             join=True)
