"""Speaker similarity using WavLM-Large and ECAPA-TDNN.

The architecture and checkpoint protocol match UniSpeech / Seed-TTS Eval.
"""

from __future__ import annotations

import os
import typing as tp
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms

from audio_eval.audio import load_audio
from audio_eval.common import AudioInput, PairedAudioCollection, paired_sources

class _Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, bias=True, scale=4):
        super().__init__()
        if channels % scale != 0:
            raise ValueError("channels must be divisible by scale")
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, stride, padding,
                      dilation, bias=bias)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x):
        outputs = []
        splits = torch.split(x, self.width, 1)
        for index in range(self.nums):
            split = splits[index] if index == 0 else split + splits[index]
            split = self.bns[index](F.relu(self.convs[index](split)))
            outputs.append(split)
        if self.scale != 1:
            outputs.append(splits[self.nums])
        return torch.cat(outputs, dim=1)


class _Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class _SEConnect(nn.Module):
    def __init__(self, channels, bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, bottleneck_dim)
        self.linear2 = nn.Linear(bottleneck_dim, channels)

    def forward(self, x):
        scale = x.mean(dim=2)
        scale = torch.sigmoid(self.linear2(F.relu(self.linear1(scale))))
        return x * scale.unsqueeze(2)


class _SERes2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, scale, bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = _Conv1dReluBn(in_channels, out_channels, 1, 1, 0)
        self.Res2Conv1dReluBn = _Res2Conv1dReluBn(
            out_channels, kernel_size, stride, padding, dilation, scale=scale
        )
        self.Conv1dReluBn2 = _Conv1dReluBn(out_channels, out_channels, 1, 1, 0)
        self.SE_Connect = _SEConnect(out_channels, bottleneck_dim)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        return self.SE_Connect(x) + residual


class _AttentiveStatsPool(nn.Module):
    def __init__(self, input_dim, attention_channels=128, global_context=False):
        super().__init__()
        self.global_context = global_context
        self.linear1 = nn.Conv1d(
            input_dim * 3 if global_context else input_dim,
            attention_channels,
            1,
        )
        self.linear2 = nn.Conv1d(attention_channels, input_dim, 1)

    def forward(self, x):
        if self.global_context:
            context_mean = x.mean(dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(x.var(dim=-1, keepdim=True) + 1e-10).expand_as(x)
            attention_input = torch.cat((x, context_mean, context_std), dim=1)
        else:
            attention_input = x
        alpha = torch.tanh(self.linear1(attention_input))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        variance = torch.sum(alpha * x.square(), dim=2) - mean.square()
        return torch.cat([mean, torch.sqrt(variance.clamp(min=1e-9))], dim=1)


class _ECAPATDNN(nn.Module):
    def __init__(self, feat_dim=1024, channels=512, emb_dim=256,
                 feat_type="wavlm_large", sample_rate=16000,
                 feature_selection="hidden_states"):
        super().__init__()
        self.feat_type = feat_type
        self.feature_selection = feature_selection
        self.sample_rate = sample_rate

        torch.hub._validate_not_a_forked_repo = lambda *args: True
        self.feature_extract = torch.hub.load("s3prl/s3prl", feat_type)
        layers = self.feature_extract.model.encoder.layers
        for index in (11, 23):
            if len(layers) > index and hasattr(layers[index].self_attn, "fp32_attention"):
                layers[index].self_attn.fp32_attention = False
        self.feat_num = self._get_feat_num()
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        freeze_names = ("final_proj", "label_embs_concat", "mask_emb", "project_q", "quantizer")
        for name, parameter in self.feature_extract.named_parameters():
            if any(value in name for value in freeze_names):
                parameter.requires_grad = False
        for parameter in self.feature_extract.parameters():
            parameter.requires_grad = False

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.channels = [channels] * 4 + [1536]
        self.layer1 = _Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = _SERes2Block(
            self.channels[0], self.channels[1], 3, 1, 2, 2, 8, 128
        )
        self.layer3 = _SERes2Block(
            self.channels[1], self.channels[2], 3, 1, 3, 3, 8, 128
        )
        self.layer4 = _SERes2Block(
            self.channels[2], self.channels[3], 3, 1, 4, 4, 8, 128
        )
        self.conv = nn.Conv1d(channels * 3, self.channels[-1], kernel_size=1)
        self.pooling = _AttentiveStatsPool(self.channels[-1], attention_channels=128)
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def _get_feat_num(self):
        self.feature_extract.eval()
        waveform = [torch.randn(self.sample_rate).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(waveform)[self.feature_selection]
        return len(features) if isinstance(features, (list, tuple)) else 1

    def _get_features(self, x):
        with torch.no_grad():
            features = self.feature_extract([sample for sample in x])[self.feature_selection]
        if isinstance(features, (list, tuple)):
            features = torch.stack(features, dim=0)
        else:
            features = features.unsqueeze(0)
        weights = F.softmax(self.feature_weight, dim=-1).view(-1, 1, 1, 1)
        features = (weights * features).sum(dim=0)
        features = torch.transpose(features, 1, 2) + 1e-6
        return self.instance_norm(features)

    def forward(self, x):
        x = self._get_features(x)
        output1 = self.layer1(x)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        output4 = self.layer4(output3)
        output = F.relu(self.conv(torch.cat([output2, output3, output4], dim=1)))
        return self.linear(self.bn(self.pooling(output)))


_MODELS: dict[tuple[str, str], _ECAPATDNN] = {}


def _checkpoint_path(checkpoint_path: str | Path | None) -> Path:
    if checkpoint_path is None:
        checkpoint_path = os.environ.get("AUDIO_EVAL_WAVLM_CHECKPOINT")
    if checkpoint_path is None:
        checkpoint_path = Path("~/.cache/audio_eval/checkpoints/wavlm_large_finetune.pth").expanduser()
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"WavLM speaker checkpoint not found at {path}. Pass checkpoint_path=... "
            "or set AUDIO_EVAL_WAVLM_CHECKPOINT."
        )
    return path


def _load_model(checkpoint_path: Path, device: str) -> _ECAPATDNN:
    key = (str(checkpoint_path.resolve()), device)
    if key not in _MODELS:
        model = _ECAPATDNN()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        _MODELS[key] = model
    return _MODELS[key]


def _embedding(model: _ECAPATDNN, source: AudioInput, sample_rate: int | None) -> torch.Tensor:
    audio, _ = load_audio(source, sample_rate=sample_rate, target_sample_rate=16000)
    waveform = torch.from_numpy(audio).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        return model(waveform)


def compute_speaker_sim(
    generated: PairedAudioCollection,
    reference: PairedAudioCollection,
    *,
    generated_sample_rate: int | None = None,
    reference_sample_rate: int | None = None,
    backend: str = "wavlm_ecapa",
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    strict: bool = True,
) -> dict:
    """Compute mean cosine similarity for one pair or two paired directories."""
    if backend != "wavlm_ecapa":
        raise ValueError(f"Unsupported speaker similarity backend: {backend!r}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = _checkpoint_path(checkpoint_path)
    model = _load_model(checkpoint, device)

    scores: list[float] = []
    details: list[dict] = []
    for key, generated_source, reference_source in paired_sources(
        generated, reference, strict=strict
    ):
        generated_embedding = _embedding(model, generated_source, generated_sample_rate)
        reference_embedding = _embedding(model, reference_source, reference_sample_rate)
        score = float(F.cosine_similarity(generated_embedding, reference_embedding).item())
        scores.append(score)
        details.append({"id": key, "speaker_sim": score})

    if not scores:
        raise ValueError("No audio pairs were evaluated for speaker similarity")
    return {
        "speaker_sim": float(np.mean(scores)),
        "backend": backend,
        "num_samples": len(scores),
        "details": details,
    }
