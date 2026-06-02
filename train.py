import argparse
import json
import random
from collections import deque
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    CLIPPromptTokenizer,
    TASBenchDataset,
    TASCollator,
    discover_ids,
    make_splits,
)
from model import GaussianDiffusion, TASDiffusionNet


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal trainable TASBench reproduction")
    parser.add_argument("--data_root", default="data/TASBench")
    parser.add_argument("--run_dir", default="code/TAS_repro/runs/tas_smoke")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--best_metric", default="val_loss", choices=["val_loss"])
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--text_model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--allow_model_download",
        action="store_true",
        help="Allow HuggingFace downloads instead of requiring cached CLIP files.",
    )
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, step: int, diffusion: GaussianDiffusion, optimizer, tokenizer, args, best_val_loss=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": diffusion.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "tokenizer_vocab": tokenizer.vocab,
            "text_model_name": tokenizer.model_name,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        path,
    )


@torch.no_grad()
def validate(diffusion: GaussianDiffusion, val_loader: DataLoader, device: torch.device, max_batches: int) -> float:
    diffusion.eval()
    losses = []
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        mono = batch["mono"].to(device, non_blocking=True)
        diff = batch["diff"].to(device, non_blocking=True)
        tokens = {key: value.to(device, non_blocking=True) for key, value in batch["tokens"].items()}
        loss = diffusion.training_loss(diff=diff, mono=mono, tokens=tokens)
        losses.append(loss.item())
    diffusion.train()
    if not losses:
        raise RuntimeError("Validation loader produced no batches.")
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    if args.fast_dev_run:
        args.max_steps = min(args.max_steps, 2)
        args.log_interval = 1
        args.save_interval = 1
        args.val_interval = 1
        args.val_batches = 1

    seed_everything(args.seed)
    data_root = Path(args.data_root)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ids = discover_ids(data_root)
    splits = make_splits(ids, seed=args.seed)
    (run_dir / f"split_seed{args.seed}.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    local_files_only = not args.allow_model_download
    tokenizer = CLIPPromptTokenizer(model_name=args.text_model_name, local_files_only=local_files_only)
    train_set = TASBenchDataset(args.data_root, splits["train"], split="train")
    val_set = TASBenchDataset(args.data_root, splits["val"], split="val")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=TASCollator(tokenizer),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=TASCollator(tokenizer),
        drop_last=False,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Training DataLoader is empty; reduce batch size or check TASBench files.")
    if len(val_loader) == 0:
        raise RuntimeError("Validation DataLoader is empty; reduce batch size or check TASBench files.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TASDiffusionNet(
        text_model_name=args.text_model_name,
        local_files_only=local_files_only,
        feature_dim=args.feature_dim,
        hidden_channels=args.hidden_channels,
    )
    diffusion = GaussianDiffusion(model, timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.lr)

    print(
        f"device={device} train_items={len(train_set)} val_items={len(val_set)} "
        f"text_model={args.text_model_name} max_steps={args.max_steps}"
    )
    print(f"run_dir={run_dir}")

    step = 0
    best_val_loss = None
    recent_losses = deque(maxlen=100)
    progress = tqdm(total=args.max_steps, desc="train", dynamic_ncols=True)
    while step < args.max_steps:
        for batch in train_loader:
            step += 1
            diffusion.train()
            mono = batch["mono"].to(device, non_blocking=True)
            diff = batch["diff"].to(device, non_blocking=True)
            tokens = {key: value.to(device, non_blocking=True) for key, value in batch["tokens"].items()}

            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.training_loss(diff=diff, mono=mono, tokens=tokens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            recent_losses.append(loss.item())

            progress.update(1)
            if step % args.log_interval == 0 or step == 1:
                train_avg = sum(recent_losses) / len(recent_losses)
                progress.set_postfix(loss=f"{loss.item():.4f}", train_avg=f"{train_avg:.4f}")
                print(f"step={step} loss={loss.item():.6f} train_loss_avg={train_avg:.6f}")

            if step % args.val_interval == 0 or step == args.max_steps:
                val_loss = validate(diffusion, val_loader, device, args.val_batches)
                train_avg = sum(recent_losses) / len(recent_losses)
                print(
                    f"step={step} train_loss_avg={train_avg:.6f} "
                    f"val_loss={val_loss:.6f} val_batches={args.val_batches}"
                )
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(run_dir / "best.pt", step, diffusion, optimizer, tokenizer, args, best_val_loss)
                    print(f"saved best checkpoint: {run_dir / 'best.pt'} val_loss={best_val_loss:.6f}")

            if step % args.save_interval == 0 or step == args.max_steps:
                save_checkpoint(run_dir / "last.pt", step, diffusion, optimizer, tokenizer, args, best_val_loss)

            if step >= args.max_steps:
                break

    progress.close()
    save_checkpoint(run_dir / "last.pt", step, diffusion, optimizer, tokenizer, args, best_val_loss)
    print(f"saved checkpoint: {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
