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

import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from nemo.collections.tts.losses.stftlosses import MultiResolutionSTFTLoss
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import AudioSignal, LossType, NormalDistributionSamplesType, VoidType
from nemo.core.neural_types.neural_type import NeuralType


window = {}
mel_basis = {}


def spectrogram(y, n_fft, hop_length, win_length, mags_min=0):
    func_sig = f"n_fft={n_fft},hop_length={hop_length},win_length={win_length},device={y.device}"
    global window
    if func_sig not in window:
        window[func_sig] = torch.hann_window(win_length).to(y.device)

    stft = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window[func_sig])
    mags = torch.sqrt(torch.clamp(stft[..., 0] ** 2 + stft[..., 1] ** 2, min=mags_min))
    return mags


def melspectrogram(y, n_fft, hop_length, win_length, sr, n_mel_channels, mel_fmin, mel_fmax, mags_min=0):
    func_sig = f"n_fft={n_fft},hop_length={hop_length},win_length={win_length},device={y.device}"
    global mel_basis
    if func_sig not in mel_basis:
        mel_basis[func_sig] = librosa_mel_fn(sr, n_fft, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis[func_sig] = torch.from_numpy(mel_basis[func_sig]).float().to(y.device)

    spec = spectrogram(y, n_fft, hop_length, win_length, mags_min)
    melspec = torch.matmul(mel_basis[func_sig], spec) 
    return melspec


def dynamic_range_compression(mag, C=1, clip_val=1e-7):
    return torch.log(torch.clamp(mag, min=clip_val) * C)


class StftLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        fft_size,
        shift_size,
        win_length,
        mel=False,
        sr=22050,
        eps=1e-6,
    ):
        """Initialize STFT loss module."""
        super(StftLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.sr = sr
        self.mel = mel
        self.eps = eps

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        if not self.mel:
            x_mag = spectrogram(x, self.fft_size, self.shift_size, self.win_length, mags_min=self.eps)
            y_mag = spectrogram(y, self.fft_size, self.shift_size, self.win_length, mags_min=self.eps)
            sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro"))
            mag_loss = F.l1_loss(torch.log(x_mag), torch.log(y_mag))
            loss = sc_loss + mag_loss
        else:
            x_mag = melspectrogram(x, self.fft_size, self.shift_size, self.win_length, self.sr, 80, 0, int(self.sr / 2))
            y_mag = melspectrogram(y, self.fft_size, self.shift_size, self.win_length, self.sr, 80, 0, int(self.sr / 2))
            x_mag, y_mag = dynamic_range_compression(x_mag), dynamic_range_compression(y_mag)
            loss = F.l1_loss(x_mag, y_mag)

        return loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
        fft_sizes=[64, 128, 256, 512, 1024, 2048, 4096],
        hop_sizes=[32, 64, 128, 256, 512, 1024, 2048],
        win_lengths=[64, 128, 256, 512, 1024, 2048, 4096],
        mel=False,
        eps=1e-6,
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [
                StftLoss(
                    fft_size=fs,
                    shift_size=ss,
                    win_length=wl,
                    mel=mel,
                    eps=eps,
                )
            ]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        loss = 0.0
        for f in self.stft_losses:
            l = f(x, y)
            loss += l
        loss /= len(self.stft_losses)

        return loss


class NllLoss(Loss):
    """A Loss module that computes loss for UniGlow"""

    @property
    def input_types(self):
        return {
            "z": NeuralType(('B', 'flowgroup', 'T'), NormalDistributionSamplesType()),
            "logdet": NeuralType(elements_type=VoidType()),
            "sigma": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, z, logdet, sigma=1.0):
        nll_loss = torch.sum(z * z) / (2 * sigma * sigma) - logdet
        nll_loss = nll_loss / (z.size(0) * z.size(1) * z.size(2))
        return nll_loss
