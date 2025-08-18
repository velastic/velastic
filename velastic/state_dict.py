from abc import ABC, abstractmethod

import bytecheckpoint as bcp
import os

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

_VELASTIC_DIR = ".velastic"


class BaseState(ABC):
    @abstractmethod
    def save(self, ckpt_path, sync=True):
        pass

    @abstractmethod
    def load(self, ckpt_path):
        pass


class BCPState(BaseState):
    def __init__(self, model, optimizer, framework, get_extra_fn=None, default_ckpt_path=None):
        self.model = model
        self.optimizer = optimizer
        self.framework = framework
        self.get_extra_fn = get_extra_fn
        self.default_ckpt_path = default_ckpt_path or os.path.join(_VELASTIC_DIR, "checkpoint")

    def state_dict(self):
        sd = {
            "model": self.model,
            "optimizer": self.optimizer,
            "extra_state": {},
        }
        if self.get_extra_fn:
            extra_state = self.get_extra_fn()
            if extra_state:
                sd["extra_state"].update(extra_state)
        return sd

    def save(self, ckpt_path=None, sync=True):
        sd = self.state_dict()
        if ckpt_path is None:
                ckpt_path = self.default_ckpt_path
        bcp.save(
            ckpt_path,
            sd,
            framework=self.framework,
            fast_saving=False if sync else True,
        )

    def load(self, ckpt_path):
        sd = self.state_dict()
        bcp.load(
            ckpt_path,
            sd,
            framework=self.framework,
            fast_loading=True,
        )
        return sd.get("extra_state", {})

    def maybe_resume_from_ckpt(self,ckpt_path=None):
        """
        Attempt to resume from the checkpoint if it exists.
        """
        try:
            if ckpt_path is None:
                ckpt_path = self.default_ckpt_path
            extra_state = self.load(ckpt_path)
            return extra_state
        except Exception as e:
            return None


class DCPState(BaseState):
    def __init__(self, model, optimizer, framework, get_extra_fn=None, default_ckpt_path=None):
        self.model = model
        self.optimizer = optimizer
        self.framework = framework
        self.get_extra_fn = get_extra_fn
        self.default_ckpt_path = default_ckpt_path or os.path.join(_VELASTIC_DIR, "dcp_checkpoint")

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "extra_state": {},
        }

    def save(self, ckpt_path=None, sync=True):
        sd = self.state_dict()
        if ckpt_path is None:
            ckpt_path = self.default_ckpt_path
        dcp.save(sd, checkpoint_id=ckpt_path)

    def load(self, ckpt_path):
        sd = self.state_dict()
        dcp.load(sd, checkpoint_id=ckpt_path)
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=sd["model"],
            optim_state_dict=sd["optimizer"]
        )
        return sd.get("extra_state", {})

    def maybe_resume_from_ckpt(self, ckpt_path=None):
        """
        Attempt to resume from the checkpoint if it exists.
        """
        try:
            if ckpt_path is None:
                ckpt_path = self.default_ckpt_path
            extra_state = self.load(ckpt_path)
            return extra_state
        except Exception as e:
            return None