import argparse
import json
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    parser.add_argument("--best_metric", default="val_total_loss", choices=["val_total_loss"])
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--tensorboard_dir", default=None)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--diff_loss_weight", type=float, default=0.1)
    parser.add_argument("--ild_loss_weight", type=float, default=0.1)
    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument(
        "--balanced_direction_sampling",
        dest="balanced_direction_sampling",
        action="store_true",
        help="Balance left/center/right prompt frames while preserving the natural mixed-direction ratio.",
    )
    balance_group.add_argument(
        "--no_balanced_direction_sampling",
        dest="balanced_direction_sampling",
        action="store_false",
        help="Disable direction-balanced prompt-frame sampling.",
    )
    parser.set_defaults(balanced_direction_sampling=True)
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
            "best_val_total_loss": best_val_loss,
            "args": vars(args),
        },
        path,
    )


class RunLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = path.open("a", encoding="utf-8")

    def write(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        self.file.write(line + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


@torch.no_grad()
def validate(diffusion: GaussianDiffusion, val_loader: DataLoader, device: torch.device, max_batches: int) -> dict:
    diffusion.eval()
    losses = defaultdict(list)
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        mono = batch["mono"].to(device, non_blocking=True)
        diff = batch["diff"].to(device, non_blocking=True)
        tokens = {key: value.to(device, non_blocking=True) for key, value in batch["tokens"].items()}
        batch_losses = diffusion.training_loss(diff=diff, mono=mono, tokens=tokens)
        for name, value in batch_losses.items():
            losses[name].append(value.item())
    diffusion.train()
    if not losses:
        raise RuntimeError("Validation loader produced no batches.")
    return {name: sum(values) / len(values) for name, values in losses.items()}


def format_loss_values(losses: dict) -> str:
    return " ".join(f"{name}={value:.6f}" for name, value in losses.items())


def direction_ratios(counts: Counter) -> dict:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {category: counts[category] / total for category in ("left", "center", "right", "mixed", "none")}


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
    log_path = Path(args.log_file) if args.log_file else run_dir / "train.log"
    tb_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else run_dir / "tensorboard"
    logger = RunLogger(log_path)
    writer = SummaryWriter(log_dir=str(tb_dir))

    ids = discover_ids(data_root)
    splits = make_splits(ids, seed=args.seed)
    (run_dir / f"split_seed{args.seed}.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    local_files_only = not args.allow_model_download
    tokenizer = CLIPPromptTokenizer(model_name=args.text_model_name, local_files_only=local_files_only)
    train_set = TASBenchDataset(
        args.data_root,
        splits["train"],
        split="train",
        balanced_direction_sampling=args.balanced_direction_sampling,
    )
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
    diffusion = GaussianDiffusion(
        model,
        timesteps=args.timesteps,
        diff_loss_weight=args.diff_loss_weight,
        ild_loss_weight=args.ild_loss_weight,
    ).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.lr)

    logger.write(
        f"device={device} train_items={len(train_set)} val_items={len(val_set)} "
        f"text_model={args.text_model_name} max_steps={args.max_steps}"
    )
    logger.write(f"run_dir={run_dir}")
    logger.write(f"tensorboard_dir={tb_dir}")
    logger.write(f"args={json.dumps(vars(args), sort_keys=True)}")
    if args.balanced_direction_sampling:
        logger.write(f"direction_sampling={json.dumps(train_set.direction_sampling_summary(), sort_keys=True)}")
    writer.add_text("run/args", json.dumps(vars(args), indent=2, sort_keys=True), global_step=0)
    writer.add_scalar("run/train_items", len(train_set), 0)
    writer.add_scalar("run/val_items", len(val_set), 0)

    step = 0
    best_val_loss = None
    recent_losses = defaultdict(lambda: deque(maxlen=100))
    sampled_directions = Counter()
    progress = tqdm(total=args.max_steps, desc="train", dynamic_ncols=True)
    while step < args.max_steps:
        for batch in train_loader:
            step += 1
            diffusion.train()
            mono = batch["mono"].to(device, non_blocking=True)
            diff = batch["diff"].to(device, non_blocking=True)
            tokens = {key: value.to(device, non_blocking=True) for key, value in batch["tokens"].items()}

            optimizer.zero_grad(set_to_none=True)
            batch_losses = diffusion.training_loss(diff=diff, mono=mono, tokens=tokens)
            loss = batch_losses["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            for name, value in batch_losses.items():
                recent_losses[name].append(value.item())
            sampled_directions.update(batch["direction_categories"])

            progress.update(1)
            if step % args.log_interval == 0 or step == 1:
                train_avg = {
                    name: sum(values) / len(values)
                    for name, values in recent_losses.items()
                }
                ratios = direction_ratios(sampled_directions)
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    train_avg=f"{train_avg['total_loss']:.4f}",
                )
                logger.write(
                    f"step={step} train_step[{format_loss_values({name: value.item() for name, value in batch_losses.items()})}] "
                    f"train_avg_100[{format_loss_values(train_avg)}] "
                    f"direction_counts={dict(sampled_directions)} direction_ratios={ratios}"
                )
                for name, value in batch_losses.items():
                    writer.add_scalar(f"loss/train_step/{name}", value.item(), step)
                for name, value in train_avg.items():
                    writer.add_scalar(f"loss/train_avg_100/{name}", value, step)
                for category, ratio in ratios.items():
                    writer.add_scalar(f"sampling/direction_ratio/{category}", ratio, step)
                    writer.add_scalar(f"sampling/direction_count/{category}", sampled_directions[category], step)
                writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], step)

            if step % args.val_interval == 0 or step == args.max_steps:
                val_losses = validate(diffusion, val_loader, device, args.val_batches)
                train_avg = {
                    name: sum(values) / len(values)
                    for name, values in recent_losses.items()
                }
                val_loss = val_losses["total_loss"]
                logger.write(
                    f"step={step} train_avg_100[{format_loss_values(train_avg)}] "
                    f"val[{format_loss_values(val_losses)}] val_batches={args.val_batches}"
                )
                for name, value in val_losses.items():
                    writer.add_scalar(f"loss/val/{name}", value, step)
                for name, value in train_avg.items():
                    writer.add_scalar(f"loss/train_avg_at_val/{name}", value, step)
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(run_dir / "best.pt", step, diffusion, optimizer, tokenizer, args, best_val_loss)
                    logger.write(f"saved best checkpoint: {run_dir / 'best.pt'} val_total_loss={best_val_loss:.6f}")
                    writer.add_scalar("loss/best_val_total_loss", best_val_loss, step)

            if step % args.save_interval == 0 or step == args.max_steps:
                save_checkpoint(run_dir / "last.pt", step, diffusion, optimizer, tokenizer, args, best_val_loss)

            if step >= args.max_steps:
                break

    progress.close()
    save_checkpoint(run_dir / "last.pt", step, diffusion, optimizer, tokenizer, args, best_val_loss)
    logger.write(f"saved checkpoint: {run_dir / 'last.pt'}")
    writer.flush()
    writer.close()
    logger.close()


if __name__ == "__main__":
    main()
