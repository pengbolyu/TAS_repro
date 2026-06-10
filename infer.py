import argparse
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from dataset import (
    CLIPPromptTokenizer,
    SAMPLE_LENGTH,
    SAMPLE_RATE,
    angle_degrees_to_direction,
    classify_prompt_direction,
    direction_to_id,
)
from model import GaussianDiffusion, TASDiffusionNet, binaural_from_mono_diff


def parse_args():
    parser = argparse.ArgumentParser(description="Generate binaural audio with TAS_repro")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--angle_degrees", type=float, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["clip", "full"], default="clip")
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--hop_sec", type=float, default=0.25)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow_model_download",
        action="store_true",
        help="Allow HuggingFace downloads instead of requiring cached CLIP files.",
    )
    return parser.parse_args()


def load_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio).transpose(0, 1)
    if waveform.shape[0] >= 2:
        mono = waveform[0:1] + waveform[1:2]
    else:
        mono = waveform[0:1]
    if sr != sample_rate:
        mono = AF.resample(mono, orig_freq=sr, new_freq=sample_rate)
    return mono.contiguous()


def crop_mono_clip(
    mono: torch.Tensor,
    start_sec: float,
    sample_rate: int = SAMPLE_RATE,
    sample_length: int = SAMPLE_LENGTH,
) -> torch.Tensor:
    total = mono.shape[-1]
    start = max(0, int(round(start_sec * sample_rate)))
    if total < sample_length:
        mono = F.pad(mono, (0, sample_length - total))
        start = 0
    elif start + sample_length > total:
        start = total - sample_length
    return mono[:, start : start + sample_length].unsqueeze(0).contiguous()


def build_window_starts(total_samples: int, sample_length: int, hop_length: int):
    if total_samples <= sample_length:
        return [0]
    starts = list(range(0, total_samples - sample_length + 1, hop_length))
    last_start = total_samples - sample_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def load_model_and_tokenizer(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    condition_type = ckpt_args.get("condition_type", "text")
    text_model_name = checkpoint.get("text_model_name") or ckpt_args.get(
        "text_model_name", "openai/clip-vit-base-patch32"
    )
    local_files_only = not args.allow_model_download
    tokenizer = None
    if condition_type == "text":
        tokenizer = CLIPPromptTokenizer(model_name=text_model_name, local_files_only=local_files_only)
    model = TASDiffusionNet(
        text_model_name=text_model_name,
        local_files_only=local_files_only,
        feature_dim=int(ckpt_args.get("feature_dim", 512)),
        hidden_channels=int(ckpt_args.get("hidden_channels", 64)),
        condition_type=condition_type,
    )
    model.load_state_dict(checkpoint["model"])
    diffusion = GaussianDiffusion(
        model,
        timesteps=int(ckpt_args.get("timesteps", 1000)),
        diff_loss_weight=float(ckpt_args.get("diff_loss_weight", 0.1)),
        ild_loss_weight=float(ckpt_args.get("ild_loss_weight", 0.1)),
    )
    return diffusion, tokenizer, condition_type


def build_conditioning(prompt: str, angle_degrees, tokenizer, condition_type: str, device: torch.device):
    if condition_type == "text":
        tokens = tokenizer.batch_encode([prompt])
        return {key: value.to(device) for key, value in tokens.items()}
    direction = angle_degrees_to_direction(float(angle_degrees)) if angle_degrees is not None else classify_prompt_direction(prompt)
    return {"direction_ids": torch.tensor([direction_to_id(direction)], dtype=torch.long, device=device)}


@torch.no_grad()
def infer_clip(
    diffusion: GaussianDiffusion,
    mono_clip: torch.Tensor,
    tokens,
    sample_steps: int,
    clip_denoised: bool,
    sampler: str,
):
    pred_diff = diffusion.sample(
        mono=mono_clip,
        tokens=tokens,
        sample_steps=sample_steps,
        clip_denoised=clip_denoised,
        sampler=sampler,
    )
    return binaural_from_mono_diff(mono_clip, pred_diff)


@torch.no_grad()
def infer_full(
    diffusion: GaussianDiffusion,
    mono: torch.Tensor,
    tokens,
    sample_steps: int,
    hop_sec: float,
    device: torch.device,
    clip_denoised: bool,
    sampler: str,
):
    original_total = mono.shape[-1]
    if original_total < SAMPLE_LENGTH:
        mono = F.pad(mono, (0, SAMPLE_LENGTH - original_total))

    hop_length = max(1, int(round(hop_sec * SAMPLE_RATE)))
    starts = build_window_starts(mono.shape[-1], SAMPLE_LENGTH, hop_length)
    diff_sum = torch.zeros(1, mono.shape[-1], device=device)
    weight_sum = torch.zeros(1, mono.shape[-1], device=device)
    window = torch.hann_window(SAMPLE_LENGTH, periodic=False, device=device).view(1, -1)
    window = window.clamp_min(1e-3)
    mono = mono.to(device)

    for index, start in enumerate(starts, start=1):
        mono_clip = mono[:, start : start + SAMPLE_LENGTH].unsqueeze(0)
        pred_diff = diffusion.sample(
            mono=mono_clip,
            tokens=tokens,
            sample_steps=sample_steps,
            clip_denoised=clip_denoised,
            sampler=sampler,
        ).squeeze(0)
        diff_sum[:, start : start + SAMPLE_LENGTH] += pred_diff * window
        weight_sum[:, start : start + SAMPLE_LENGTH] += window
        print(f"window {index}/{len(starts)} start={start / SAMPLE_RATE:.2f}s")

    full_diff = diff_sum / weight_sum.clamp_min(1e-6)
    full_diff = full_diff[:, :original_total]
    full_mono = mono[:, :original_total]
    return binaural_from_mono_diff(full_mono.unsqueeze(0), full_diff.unsqueeze(0))


def main():
    args = parse_args()
    if args.hop_sec <= 0:
        raise ValueError("--hop_sec must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    diffusion, tokenizer, condition_type = load_model_and_tokenizer(args)
    diffusion = diffusion.to(device)
    diffusion.eval()

    mono = load_mono(Path(args.input_wav))
    tokens = build_conditioning(args.prompt, args.angle_degrees, tokenizer, condition_type, device)

    if args.mode == "clip":
        mono_clip = crop_mono_clip(mono, args.start_sec).to(device)
        binaural = infer_clip(
            diffusion,
            mono_clip,
            tokens,
            args.sample_steps,
            args.clip_denoised,
            args.sampler,
        )
    else:
        print(
            f"full inference duration={mono.shape[-1] / SAMPLE_RATE:.2f}s "
            f"window={SAMPLE_LENGTH / SAMPLE_RATE:.2f}s hop={args.hop_sec:.2f}s "
            f"sample_steps={args.sample_steps} sampler={args.sampler}"
        )
        binaural = infer_full(
            diffusion,
            mono,
            tokens,
            args.sample_steps,
            args.hop_sec,
            device,
            args.clip_denoised,
            args.sampler,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav = binaural.squeeze(0).detach().cpu().transpose(0, 1).numpy()
    peak = abs(wav).max()
    if peak > 1.0:
        print(f"warning: output peak {peak:.4f} exceeds 1.0; clipping before wav write")
        wav = wav.clip(-1.0, 1.0)
    sf.write(str(out_path), wav, SAMPLE_RATE)
    print(f"saved: {out_path} sr={SAMPLE_RATE} channels=2 samples={wav.shape[0]}")


if __name__ == "__main__":
    main()
