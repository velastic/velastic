# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from transformers import BatchFeature, PretrainedConfig, PreTrainedModel, ProcessorMixin


class BaseEncoderConfigMixin(PretrainedConfig, ABC):
    def __init__(self, output_size=None, add_projector=False, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.add_projector = add_projector
        self.initializer_range = initializer_range


class BaseEncoderProcessorMixin(ProcessorMixin, ABC):
    # TODO: Fix: Some kwargs in processor config are unused and will not have any effect
    attributes = []
    optional_attributes = ["chat_template"]

    def __init__(self, token_size=None, token_num=None, **kwargs):
        """ """
        super_init_kwargs = {}
        for key in kwargs.keys():
            if key in self.attributes + self.optional_attributes:
                super_init_kwargs[key] = kwargs[key]
        super().__init__(**super_init_kwargs)
        self.token_size = token_size
        self.token_num = token_num

    @abstractmethod
    def process(self, **kwargs) -> BatchFeature:
        """ """
        pass


class BaseEncoderModelMixin(PreTrainedModel):
    @abstractmethod
    def set_projector_trainable_only(self) -> None:
        """Sets only the projector layers to be trainable."""
        pass

    @abstractmethod
    def _get_lm_dummy_data(self) -> Dict[str, torch.Tensor]:
        """ """
        pass

    @abstractmethod
    def lm_encode(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """ """
        pass

    def lm_dummy_encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "_dummy_data", None) is None:
            self._dummy_data = self._get_lm_dummy_data()
        return self.lm_encode(**self._dummy_data)
