# velastic

## Installation
```bash
pip install -e .
```

## Module Explanation

velastic consists of two core modules:

1. **State**: Responsible for saving and loading model and optimizer states.
2. **IterManager**: Responsible for managing training iterations, including statistics and prediction of iteration times, and triggering checkpoint saves at appropriate times.

For frameworks that already have a complete save/load checkpoint solution, only the IterManager module needs to be integrated.

## Integrating velastic into a Framework

### Integrating Both State and IterManager (using nanoGPT as an example)

In the training loop, the following modifications are needed:
1. Define get_extra_state()
```python
def get_extra_state():
    return {
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "X": X,
        "Y": Y,
        "scaler_state": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
```


2. Initialize State and IterManager:

```python
from velastic.manager import IterManager
from velastic.state_dict import BCPState

# Initialize velastic modules
state = BCPState(model=model, optimizer=optimizer, framework="ddp", get_extra_fn=get_extra_state)
iter_mngr = IterManager(total_iters=max_iters, use_torch_compile=compile)
```

3. Add resume logic
```python
extra_state = state.maybe_resume_from_ckpt()
if extra_state is not None:
    # Restore extra state
    iter_num = extra_state['iter_num']
    best_val_loss = extra_state['best_val_loss']
    # Restore other states
    X = extra_state['X']
    Y = extra_state['Y']
    scaler.load_state_dict(extra_state['scaler_state'])
    torch.set_rng_state(extra_state['torch_rng_state'])
    torch.cuda.set_rng_state(extra_state['cuda_rng_state'])
```

4. Add velastic-related calls in the training loop:

```python
while True:
    # Call at the beginning of each iteration
    iter_mngr.iter()
    
    # Save checkpoint based on IterManager's suggestion
    if iter_mngr.should_save_ckpt():
        state.save(sync=False)
    
    # Other training code...
```

### Integrating Only IterManager (for frameworks with existing save/load checkpoint solutions)

For frameworks that already have a complete checkpoint solution, you can directly use IterManager's `should_save_ckpt()` method to decide when to save checkpoints:

```python
from velastic.manager import IterManager

# Initialize IterManager
iter_mngr = IterManager(total_iters=max_iters, use_torch_compile=compile)

# In the training loop
while True:
    # Call at the beginning of each iteration
    iter_mngr.iter()
    
    # Save checkpoint based on IterManager's suggestion
    if iter_mngr.should_save_ckpt():
        # Call the framework's existing checkpoint saving logic
        save_checkpoint()
    
    # Other training code...
```

## Usage
For example, to run the nanoGPT training script with velastic:
```bash
cd examples/nanoGPT
# First prepare the dataset refer to the README.md in the examples/nanoGPT directory
python data/shakespeare_char/prepare.py

# Simulate ESI for the training job
python ../../esi_simulate.py --cmd 'torchrun --standalone --nproc-per-node 2 train.py config/train_shakespeare_char.py' --timeout 120