import argparse
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from dataset import CLIPPromptTokenizer, SAMPLE_LENGTH, SAMPLE_RATE
from model import GaussianDiffusion, TASDiffusionNet, binaural_from_mono_diff


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a 1-second binaural sample with TAS_repro")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow_model_download",
        action="store_true",
        help="Allow HuggingFace downloads instead of requiring cached CLIP files.",
    )
    return parser.parse_args()


def load_mono_clip(path: Path, start_sec: float, sample_rate: int = SAMPLE_RATE, sample_length: int = SAMPLE_LENGTH):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio).transpose(0, 1)
    if waveform.shape[0] >= 2:
        mono = waveform[0:1] + waveform[1:2]
    else:
        mono = waveform[0:1]
    if sr != sample_rate:
        mono = AF.resample(mono, orig_freq=sr, new_freq=sample_rate)

    total = mono.shape[-1]
    start = max(0, int(round(start_sec * sample_rate)))
    if total < sample_length:
        mono = F.pad(mono, (0, sample_length - total))
        start = 0
    elif start + sample_length > total:
        start = total - sample_length
    return mono[:, start : start + sample_length].unsqueeze(0).contiguous().clamp(-1.0, 1.0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    text_model_name = checkpoint.get("text_model_name") or ckpt_args.get("text_model_name", "openai/clip-vit-base-patch32")
    local_files_only = not args.allow_model_download
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = CLIPPromptTokenizer(model_name=text_model_name, local_files_only=local_files_only)
    model = TASDiffusionNet(
        text_model_name=text_model_name,
        local_files_only=local_files_only,
        feature_dim=int(ckpt_args.get("feature_dim", 512)),
        hidden_channels=int(ckpt_args.get("hidden_channels", 64)),
    )
    model.load_state_dict(checkpoint["model"])
    diffusion = GaussianDiffusion(model, timesteps=int(ckpt_args.get("timesteps", 1000))).to(device)
    diffusion.eval()

    mono = load_mono_clip(Path(args.input_wav), args.start_sec).to(device)
    tokens = tokenizer.batch_encode([args.prompt])
    tokens = {key: value.to(device) for key, value in tokens.items()}

    with torch.no_grad():
        pred_diff = diffusion.sample(mono=mono, tokens=tokens, sample_steps=args.sample_steps)
        binaural = binaural_from_mono_diff(mono, pred_diff)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav = binaural.squeeze(0).detach().cpu().transpose(0, 1).numpy()
    sf.write(str(out_path), wav, SAMPLE_RATE)
    print(f"saved: {out_path} sr={SAMPLE_RATE} channels=2 samples={wav.shape[0]}")


if __name__ == "__main__":
    main()
