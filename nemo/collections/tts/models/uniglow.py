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

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import time
import torch
from torchaudio.transforms import MFCC
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from pystoi import stoi
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger

from nemo.collections.tts.helpers.helpers import OperationMode, waveglow_log_to_tb_func
from nemo.collections.tts.models.base import Vocoder
from nemo.collections.tts.modules.uniglow import UniGlowModule
from nemo.collections.tts.modules.gan_modules import MultiScaleDiscriminator, MultiPeriodDiscriminator, discriminator_loss, generator_loss, feature_loss
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    AudioSignal,
    LengthsType,
    LogDeterminantType,
    MelSpectrogramType,
    NormalDistributionSamplesType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging


class UniGlowModel(Vocoder):
    """UniGlow model used to convert betweeen spectrograms and audio"""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        self.removed_weightnorm = False
        self.sigma = self._cfg.sigma
        self.audio_to_melspec_precessor = instantiate(self._cfg.preprocessor)
        self.model = UniGlowModule(
            self._cfg.n_mel_channels,
            self._cfg.n_flows,
            self._cfg.n_group,
            self._cfg.n_wn_channels,
            self._cfg.n_wn_layers,
            self._cfg.wn_kernel_size,
            self.get_upsample_factor(),
        )

        self.nll_loss = instantiate(cfg.nll_loss)
        self.stft_loss = instantiate(cfg.stft_loss)
        
        if cfg.adv_loss_coef > 0:
            self.msd = MultiScaleDiscriminator()
            self.mpd = MultiPeriodDiscriminator()

    @property
    def input_types(self):
        return {
            "spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "sigma": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'S', 'T'), AudioSignal(self.sample_rate)),
        }

    @typecheck()
    def forward(self, *, spec: torch.Tensor, sigma: float = 1.0):
        return self.model.infer(spec=spec, sigma=sigma)

    @typecheck(
        input_types={"spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()), "sigma": NeuralType(optional=True)},
        output_types={"audio": NeuralType(('B', 'T'), AudioSignal())},
    )
    def convert_spectrogram_to_audio(self, spec: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        if not self.removed_weightnorm:
            self.waveglow.remove_weightnorm()
            self.removed_weightnorm = True
        audio = self.infer(spec=spec, audio=None, sigma=sigma)
        return audio

    def training_step(self, batch, batch_idx):
        audio, audio_len = batch
        spec, spec_len = self.audio_to_melspec_precessor(audio, audio_len)

        z, logdet = self.model(spec=spec, audio=audio)
        predicted_audio = self.model.infer(spec=spec, sigma=self.sigma)

        nll_loss = self.nll_loss(z=z, logdet=logdet, sigma=self.sigma)
        stft_loss = self.stft_loss(x=predicted_audio, y=audio)

        loss = self._cfg.nll_loss_coef * nll_loss + \
               self._cfg.stft_loss_coef * stft_loss

        metrics = {
            "nll_loss": nll_loss,
            "stft_loss": stft_loss,
        }

        # vocoder adversarial loss term
        if self._cfg.adv_loss_coef > 0:
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(audio.unsqueeze(1), predicted_audio.unsqueeze(1))
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(audio.unsqueeze(1), predicted_audio.unsqueeze(1))
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_f + loss_gen_s
            if self._cfg.fm_loss:
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_fm = loss_fm_s + loss_fm_f
                loss_gen_all += loss_fm
                metrics["fm_loss"] = loss_fm
            loss += self._cfg.adv_loss_coef * loss_gen_all
            metrics["gen_loss"] = loss_gen_all

        self.manual_backward(loss, self.optimizer)
        self.manual_optimizer_step(self.optimizer)

        if self._cfg.adv_loss_coef > 0:
            # mpd
            y_df_hat_r, y_df_hat_g, _, _ = self.mpd(audio.unsqueeze(1), predicted_audio.detach().unsqueeze(1))
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            # msd
            y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(audio.unsqueeze(1), predicted_audio.detach().unsqueeze(1))
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            # compute overall discriminator loss
            d_loss = loss_disc_s + loss_disc_f

            metrics.update({
                "msd_loss": loss_disc_s,
                "mpd_loss": loss_disc_f,
                "d_loss": d_loss
            })
            self.manual_backward(d_loss, self.d_optimizer)
            self.manual_optimizer_step(self.d_optimizer)

        self.log_dict(metrics, on_step=False, on_epoch=True)
        return {}

    def validation_step(self, batch, batch_idx):
        audio, audio_len = batch
        spec, spec_len = self.audio_to_melspec_precessor(audio, audio_len)

        z, logdet = self.model(spec=spec, audio=audio)
        predicted_audio = self.model.infer(spec=spec, sigma=self.sigma)

        sr = self._cfg.preprocessor.sample_rate
        nll_loss = self.nll_loss(z=z, logdet=logdet, sigma=self.sigma)
        stft_loss = self.stft_loss(x=predicted_audio, y=audio)
        stoi_score = np.mean([stoi(a.cpu(),b.cpu(),sr) for a,b in zip(audio, predicted_audio)])
        mfcc_fn = MFCC(sample_rate=sr).to(spec.device)
        mfcc_distance = torch.mean(torch.abs(mfcc_fn(audio) - mfcc_fn(predicted_audio)))

        metrics = {
            f"val_nll_loss": nll_loss,
            f"val_stft_loss": stft_loss,
            f"val_stoi": torch.FloatTensor([stoi_score]),
            f"val_mfcc_distance": mfcc_distance
        }
        self.log_dict(metrics)

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizers = []
        self.optimizer = instantiate(self._cfg.optimizer, params=self.model.parameters())
        if self._cfg.adv_loss_coef > 0:
            d_params = list(self.msd.parameters()) + list(self.mpd.parameters())
            self.d_optimizer = instantiate(self._cfg.optimizer, params=d_params)

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg["dataloader_params"]):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="UniGlow-22050Hz",
            location="https://drive.google.com/file/d/18JO5heoz1pBicZnGGqJzAJYMpzxiDQDa/view?usp=sharing",
            description="The model is trained on LJSpeech sampled at 22050Hz, and can be used as an universal vocoder",
        )
        list_of_models.append(model)
        return list_of_models

    def get_upsample_factor(self) -> int:
        """
        As the MelSpectrogram upsampling is done using interpolation, the upsampling factor is determined
        by the ratio of the MelSpectrogram length and the waveform length
        Returns:
            An integer representing the upsampling factor
        """
        audio = torch.ones(1, self._cfg.train_ds.dataset.n_segments)
        spec, spec_len = self.audio_to_melspec_precessor(audio, torch.FloatTensor([len(audio)]))
        spec = spec[:, :, :-1]
        audio = audio.unfold(1, self._cfg.n_group, self._cfg.n_group).permute(0, 2, 1)
        upsample_factor = audio.shape[2] // spec.shape[2]
        return upsample_factor
