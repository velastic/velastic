import json
import logging
import os
import shutil
import time
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
import numpy as np

import torch.distributed as dist

from .state_dict import BaseState

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("bytecheckpoint").setLevel(logging.WARNING)

_VELASTIC_STATE_FILENAME = "manager_state.json"
_VELASTIC_BIN_FILENAME = "manager_state.pkl"


class IterManager:
    def __init__(
        self,
        total_iters: int = -1,
        redundant_time: float = 5,
        save_ckpt_time: float = 0,
        use_torch_compile: bool = False,
        predict_freq: int = 5,
        velastic_dir: str = ".velastic",
    ):
        self.total_iters = total_iters
        self.redundant_time = max(redundant_time, 0.0)
        self.use_torch_compile = use_torch_compile
        self.save_ckpt_time = save_ckpt_time
        self.predict_freq = predict_freq
        # Initialize iteration tracking
        self.iter_times = []
        self.start_time = None
        self.iter_count = 0
        
        # Directory and state file for Velastic
        self.velastic_dir = velastic_dir
        self.velastic_state_path = os.path.join(self.velastic_dir, _VELASTIC_STATE_FILENAME)
        self.velastic_bin_path = os.path.join(self.velastic_dir, _VELASTIC_BIN_FILENAME)

    def iter(self):
        """
        Generator-style iteration method
        Replaces context manager pattern to avoid indentation changes
        """
        is_resuming = False
        if os.path.exists(self.velastic_bin_path) and self.iter_count == 0:
            self.load_manager_state()
            os.remove(self.velastic_bin_path)
            is_resuming = True
        new_time = time.perf_counter()
        if self.iter_count > 0 and not is_resuming:
            duration = new_time - self.start_time
            self.iter_times.append(duration)
        # if self.iter_count > 0:
        #     duration = new_time - self.start_time
        #     self.iter_times.append(duration)
        self.iter_count += 1
        self.start_time = new_time
        if self.total_iters > 0 and self.iter_count >= self.total_iters:
            self.cleanup()
        else:
            self.save_manager_state()


    @property
    def avg_iter_time(self):
        """
        Calculate the average iteration time.
        If using torch compile, skip the first iteration time as it is usually longer.
        """
        if self.iter_times is None or len(self.iter_times) == 0:
            return 0.0

        iter_times = self.iter_times
        # Skip the first iteration if using torch compile
        if self.use_torch_compile and len(self.iter_times) > 1:
            iter_times = self.iter_times[1:]
        return sum(iter_times) / len(iter_times)

    @property
    def max_iter_time(self):
        """
        Get the maximum iteration time.
        """
        if self.iter_times is None or len(self.iter_times) == 0:
            return 0.0

        iter_times = self.iter_times
        if self.use_torch_compile and len(self.iter_times) > 1:
            iter_times = self.iter_times[1:]
        return max(iter_times)

    def should_save_ckpt(self):

        exp_ts = os.getenv("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP", None)
        # no need to save if the resource is not going to expire
        if exp_ts is None:
            return False
        res_remain_time = float(exp_ts) - time.time()

        if res_remain_time < self.save_ckpt_time + self.redundant_time +  self.max_iter_time:
            self.save_manager_state(save_bin=True)
            return True
        return False
    
    def save_manager_state(self,save_bin=False):
        """
        Save the manager state to '.velastic/manager_state.json'.
        This includes average iteration time, total iterations, iteration count, and estimated remaining time.
        """
        # Only save from rank 0 in distributed training
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        # Save the manager state to a JSON file
        os.makedirs(self.velastic_dir, exist_ok=True)
        if(self.total_iters > 0):
            remaining_time=self.avg_iter_time * (self.total_iters - self.iter_count) 
        else:
            remaining_time = -1
        manager_state = {
            "avg_iter_time": self.avg_iter_time,
            "total_iters": self.total_iters,
            "iter_count": self.iter_count,
            "remaining_time": remaining_time,
            "save_ckpt_time": self.save_ckpt_time,
        }
        with open(self.velastic_state_path, "w") as f:
            json.dump(manager_state, f, indent=4)
            
        # Save the entire IterManager object as binary file
        if save_bin:
            with open(self.velastic_bin_path, "wb") as f:
                pickle.dump(self, f)

    def load_manager_state(self):
        """
        Load from '.velastic/manager_state.json' and set IterManager's iteration state.
        """
        try:
            # Load the entire IterManager object from binary file
            if os.path.exists(self.velastic_bin_path):
                with open(self.velastic_bin_path, "rb") as f:
                    loaded_manager = pickle.load(f)
                    # Restore the state of current object
                    self.__dict__.update(loaded_manager.__dict__)
            else:
                # Fallback to JSON loading if binary file doesn't exist
                with open(self.velastic_state_path) as f:
                    manager_state = json.load(f)
                self.iter_count = manager_state.get("iter_count", self.iter_count)
                self.save_ckpt_time = manager_state.get("save_ckpt_time", self.save_ckpt_time)
        except Exception as e:
            logger.error(f"Failed to load manager state from {self.velastic_state_path}: {e}")
            raise


    def cleanup(self):
        """
        Cleanup saved state.
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        def safe_remove(path):
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

        safe_remove(self.velastic_dir)
        logger.info("Cleanup completed, removed saved state.")