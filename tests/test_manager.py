import os
import json
import time
import shutil
from unittest.mock import patch, MagicMock

import pytest
from velastic.manager import IterManager, _VELASTIC_STATE_FILENAME

test_dir = "/tmp/.test_velastic"
@pytest.fixture
def manager():
    """ Fixture to create an IterManager instance for testing. """
    # Use a temporary directory for velastic files
    
    os.environ["MLP_TASK_VELASTIC_STATE_DIR"] = test_dir
    manager = IterManager()
    yield manager

def test_initialization(manager):
    """ Test the initialization of IterManager. """
    assert manager.total_iters == -1
    assert manager.redundant_time == 5
    assert manager.save_ckpt_time == 0
    assert not manager.use_torch_compile
    assert manager.predict_freq == 5
    assert manager.iter_count == 0
    assert manager.velastic_dir == test_dir
    assert os.path.exists(manager.velastic_dir)

def test_iter_method(manager):
    """ Test the iter method. """
    manager.iter()
    assert manager.iter_count == 1
    assert manager.start_time is not None

    time.sleep(0.1)
    manager.iter()
    assert manager.iter_count == 2
    assert len(manager.iter_times) == 1
    assert manager.iter_times[0] > 0

def test_avg_iter_time(manager):
    """ Test the avg_iter_time property. """
    manager.iter_times = [0.1, 0.2, 0.3]
    assert manager.avg_iter_time == pytest.approx(0.2)

def test_avg_iter_time_with_torch_compile(manager):
    """ Test avg_iter_time with torch_compile enabled. """
    manager.use_torch_compile = True
    manager.iter_times = [1.0, 0.1, 0.2, 0.3]
    assert manager.avg_iter_time == pytest.approx(0.2)

def test_max_iter_time(manager):
    """ Test the max_iter_time property. """
    manager.iter_times = [0.1, 0.3, 0.2]
    assert manager.max_iter_time == 0.3

def test_save_and_load_manager_state_json(manager):
    """ Test saving and loading the manager state using JSON. """
    manager.iter_count = 10
    manager.save_ckpt_time = 100
    with patch('torch.distributed.is_initialized', return_value=False):
        manager.save_manager_state()

    # Create a new manager to load the state
    new_manager = IterManager(velastic_dir=manager.velastic_dir)
    new_manager.load_manager_state()

    assert new_manager.iter_count == 10
    assert new_manager.save_ckpt_time == 100

def test_save_and_load_manager_state_pickle(manager):
    """ Test saving and loading the manager state using pickle. """
    manager.iter_count = 20
    manager.iter_times = [0.1, 0.2]
    with patch('torch.distributed.is_initialized', return_value=False):
        manager.save_manager_state(save_bin=True)

    # Create a new manager to load the state
    new_manager = IterManager(velastic_dir=manager.velastic_dir)
    new_manager.load_manager_state()

    assert new_manager.iter_count == 20
    assert new_manager.iter_times == [0.1, 0.2]

@patch('torch.distributed.is_initialized', return_value=False)
def test_should_save_ckpt(mock_dist, manager):
    """ Test the should_save_ckpt method. """
    # Case 1: No expiration timestamp, should not save
    with patch('os.getenv', return_value=None):
        assert not manager.should_save_ckpt()

    # Case 2: Remaining time is greater than threshold, should not save
    future_time = time.time() + 200
    with patch('os.getenv', return_value=str(future_time)):
        manager.save_ckpt_time = 60
        manager.redundant_time = 10
        manager.iter_times = [1]
        assert not manager.should_save_ckpt()

    # Case 3: Remaining time is less than threshold, should save
    future_time = time.time() + 70
    with patch('os.getenv', return_value=str(future_time)):
        manager.save_ckpt_time = 60
        manager.redundant_time = 10
        manager.iter_times = [1]
        assert manager.should_save_ckpt()
        # Check if the binary state was saved
        assert os.path.exists(manager.velastic_bin_path)

def test_cleanup(manager):
    """ Test the cleanup method. """
    with patch('torch.distributed.is_initialized', return_value=False):
        manager.save_manager_state()
        assert os.path.exists(manager.velastic_dir)
        manager.cleanup()
        assert not os.path.exists(manager.velastic_dir)

def test_resume_from_state(manager):
    """ Test resuming from a saved binary state. """
    manager.iter_count = 50
    manager.iter_times = [0.5] * 50
    with patch('torch.distributed.is_initialized', return_value=False):
        manager.save_manager_state(save_bin=True)

    # Simulate a restart by creating a new manager and calling iter
    new_manager = IterManager(velastic_dir=manager.velastic_dir)
    new_manager.iter() # This should load the state

    assert new_manager.iter_count == 51 # 50 from loaded state + 1 from iter()
    assert len(new_manager.iter_times) == 50
    assert not os.path.exists(new_manager.velastic_bin_path) # binary file should be removed after loading