import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device).float() / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 512,
        local_files_only: bool = True,
    ):
        super().__init__()
        try:
            from transformers import CLIPTextModel
        except ImportError as exc:
            raise ImportError(
                "CLIP text guidance requires transformers. Install it in the HRTF environment, "
                "for example: F:\\Anaconda\\envs\\HRTF\\python.exe -m pip install transformers"
            ) from exc
        self.encoder = CLIPTextModel.from_pretrained(model_name, local_files_only=local_files_only)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens.get("attention_mask"),
            )
            text_features = outputs.pooler_output
        return self.norm(text_features)


class AudioEncoder(nn.Module):
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, feature_dim, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=5, padding=2),
            nn.SiLU(),
        )

    def forward(self, mono: torch.Tensor) -> torch.Tensor:
        return self.net(mono)


class SemanticAwareFusion(nn.Module):
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
        )

    def forward(self, audio_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(audio_features.shape[1])
        scores = (audio_features * text_features.unsqueeze(-1)).sum(dim=1) / scale
        weights = torch.softmax(scores, dim=-1)
        text_aware_audio = (audio_features * weights.unsqueeze(1)).sum(dim=-1)
        fused_vector = text_features * (self.fc(text_aware_audio) + text_aware_audio)
        repeated = fused_vector.unsqueeze(-1).expand(-1, -1, audio_features.shape[-1])
        return self.conv(torch.cat([audio_features, repeated], dim=1))


class DilatedResidualBlock(nn.Module):
    def __init__(self, hidden_channels: int, cond_channels: int, time_dim: int, layers: int = 10):
        super().__init__()
        self.cond_proj = nn.Conv1d(cond_channels, hidden_channels, kernel_size=1)
        self.time_proj = nn.Linear(time_dim, hidden_channels)
        self.layers = nn.ModuleList()
        for i in range(layers):
            dilation = 2**i
            self.layers.append(
                nn.Sequential(
                    nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
                    nn.SiLU(),
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                )
            )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = x + self.cond_proj(cond) + self.time_proj(time_emb).unsqueeze(-1)
        for layer in self.layers:
            h = h + layer(h)
        return h


class TASDiffusionNet(nn.Module):
    def __init__(
        self,
        text_model_name: str = "openai/clip-vit-base-patch32",
        local_files_only: bool = True,
        feature_dim: int = 512,
        hidden_channels: int = 64,
        time_dim: int = 128,
        residual_blocks: int = 3,
        dilated_layers: int = 10,
    ):
        super().__init__()
        self.text_encoder = CLIPTextEncoder(
            model_name=text_model_name,
            embed_dim=feature_dim,
            local_files_only=local_files_only,
        )
        self.audio_encoder = AudioEncoder(feature_dim=feature_dim)
        self.saf = SemanticAwareFusion(feature_dim=feature_dim)
        self.input_proj = nn.Conv1d(2, hidden_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    hidden_channels=hidden_channels,
                    cond_channels=feature_dim,
                    time_dim=time_dim,
                    layers=dilated_layers,
                )
                for _ in range(residual_blocks)
            ]
        )
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels * residual_blocks),
            nn.SiLU(),
            nn.Conv1d(hidden_channels * residual_blocks, 1, kernel_size=3, padding=1),
        )
        self.time_dim = time_dim

    def forward(self, noisy_diff: torch.Tensor, mono: torch.Tensor, tokens: torch.Tensor, timesteps: torch.Tensor):
        text_features = self.text_encoder(tokens)
        audio_features = self.audio_encoder(mono)
        cond = self.saf(audio_features, text_features)
        time_emb = self.time_mlp(timestep_embedding(timesteps, self.time_dim))

        h = self.input_proj(torch.cat([noisy_diff, mono], dim=1))
        outputs = []
        for block in self.blocks:
            h = block(h, cond, time_emb)
            outputs.append(h)
        return self.out(torch.cat(outputs, dim=1))


class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def training_loss(self, diff: torch.Tensor, mono: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch = diff.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch,), device=diff.device)
        noise = torch.randn_like(diff)
        noisy_diff = self.q_sample(diff, timesteps, noise)
        pred_noise = self.model(noisy_diff, mono, tokens, timesteps)
        return F.mse_loss(pred_noise, noise)
