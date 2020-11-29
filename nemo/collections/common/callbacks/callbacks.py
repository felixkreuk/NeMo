# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import time
import torch

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only

from nemo.utils import logging


class LogEpochTimeCallback(Callback):
    """Simple callback that logs how long each epoch takes, in seconds, to a pytorch lightning log
    """

    @rank_zero_only
    def on_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    @rank_zero_only
    def on_epoch_end(self, trainer, pl_module):
        curr_time = time.time()
        duration = curr_time - self.epoch_start
        trainer.logger.log_metrics({"epoch_time": duration}, step=trainer.global_step)


class StdoutMetricLogger(Callback):
    """A callback to print a summary of metrics on_epoch_end"""

    @rank_zero_only
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        processed_metrics = {}
        for k,v in metrics.items():
            if isinstance(v, torch.Tensor):
                processed_metrics[k] = v.item()
            else:
                processed_metrics[k] = v

        processed_metrics["epoch"] = pl_module.current_epoch + 1
        processed_metrics["global_step"] = pl_module.global_step + 1

        logging.info("")
        logging.info("EPOCH SUMMARY:")
        logging.info("==============")
        for k,v in processed_metrics.items():
            logging.info(f"{k} = {v:.4f}")
        logging.info("")
