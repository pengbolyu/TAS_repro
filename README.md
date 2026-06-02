# TASBench minimal training reproduction

This is a small trainable reproduction scaffold for **TAS: Personalized Text-guided Audio Spatialization**.

It is designed to run the training path only. It uses a frozen `openai/clip-vit-base-patch32` text encoder and does not include paper metrics, inference overlap-add, or baseline comparisons.

The HRTF environment must have `transformers` installed. By default, CLIP is loaded from the local HuggingFace cache to avoid network retries. Add `--allow_model_download` only when the model is not cached yet.

## Fast smoke run

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\train.py --data_root data\TASBench --batch_size 1 --fast_dev_run
```

## Normal training start

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\train.py --data_root data\TASBench --batch_size 12 --max_steps 3000 --lr 2e-4
```

If CUDA memory is insufficient, reduce `--batch_size` or `--hidden_channels`.
