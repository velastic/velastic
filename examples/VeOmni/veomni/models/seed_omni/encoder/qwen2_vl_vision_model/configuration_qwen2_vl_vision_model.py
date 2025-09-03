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

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig

from ..base import BaseEncoderConfigMixin


class Qwen2VLVisionModelConfig(BaseEncoderConfigMixin, Qwen2VLVisionConfig):
    model_type = "qwen2_vl_vision_model"

    def __init__(
        self,
        return_hidden_states=False,
        **kwargs,
    ):
        self.return_hidden_states = return_hidden_states
        super().__init__(**kwargs)
