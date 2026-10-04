"""Generate auditable conditional 3D samples from one checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

from demo_diffusion import dist_util, logger
from demo_diffusion.demo_image_datasets import read_binary_tiff
from demo_diffusion.full_model import require_porosity_condition
from demo_diffusion.demo_script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from demo_diffusion.reproducibility import (
    seed_everything,
    write_model_summary,
    write_run_metadata,
)


def _sample_fields():
    return [
        "sample_index",
        "seed",
        "seed_scope",
        "target_porosity",
        "generated_porosity",
        "absolute_porosity_error",
        "sampler",
        "runtime_seconds",
        "binary_path",
        "raw_path",
    ]


def main():
    args = create_argparser().parse_args()
    require_porosity_condition(args.use_condition)
    checkpoint = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"--model_path does not exist: {checkpoint}")
    if not 0.0 <= args.target_porosity <= 1.0:
        raise ValueError("--target_porosity must be within [0, 1]")
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("--num_samples and --batch_size must be positive")

    dist_util.setup_dist()
    if dist_util.get_world_size() != 1:
        raise RuntimeError(
            "demo_sample_num.py currently supports one sampling process only; "
            "launch independent explicit seeds into distinct output directories"
        )
    seed_everything(args.seed, deterministic=args.deterministic)
    logger.configure(dir=str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        shape = tuple(
            int(value)
            for value in read_binary_tiff(
                args.data_dir,
                threshold=args.binary_threshold,
                pore_is_high=args.pore_is_high,
            ).shape
        )
    else:
        shape = (args.image_size, args.image_size, args.image_size)
    divisor = 2 ** max(0, len(str(args.channel_mult).split(",")) - 1)
    if any(dimension % divisor for dimension in shape):
        raise ValueError(
            f"sample shape {shape} must be divisible by model downsampling "
            f"factor {divisor}"
        )

    if dist_util.get_rank() == 0:
        run_config = dict(vars(args))
        run_config["target_porosity"] = float(args.target_porosity)
        run_config.update(
            resolved_model_path=str(checkpoint),
            resolved_output_dir=str(output_dir),
            resolved_data_dir=(
                str(Path(args.data_dir).expanduser().resolve())
                if args.data_dir
                else None
            ),
            sample_shape=list(shape),
            phase_convention={"pore": 1.0, "grain": -1.0},
        )
        write_run_metadata(
            output_dir,
            run_config,
            run_type="sample",
            repository_root=Path(__file__).resolve().parent,
        )

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state_dict = dist_util.load_state_dict(str(checkpoint), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(dist_util.dev())
    if dist_util.get_rank() == 0:
        write_model_summary(
            output_dir, model, requested_attention=args.attention_resolutions
        )
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    sampler_name = "ddim" if args.use_ddim else "ddpm"
    sample_fn = (
        diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
    )
    log_path = output_dir / "sampling_log.csv"
    if dist_util.get_rank() == 0:
        with log_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=_sample_fields()).writeheader()

    conditioning_description = f"target_porosity={args.target_porosity}"
    logger.log(
        f"sampling {args.num_samples} volume(s), shape={shape}, "
        f"{conditioning_description}, sampler={sampler_name}"
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    total_sampling_seconds = 0.0
    completed = 0
    while completed < args.num_samples:
        current_batch = min(args.batch_size, args.num_samples - completed)
        batch_seed = args.seed + completed
        seed_everything(batch_seed, deterministic=args.deterministic)
        labels = torch.full(
            (current_batch, 1),
            float(args.target_porosity),
            device=dist_util.dev(),
        )
        noise = torch.randn(
            (current_batch, 1, *shape), device=dist_util.dev()
        )
        started = time.perf_counter()
        with torch.no_grad():
            sample = sample_fn(
                model,
                (current_batch, 1, *shape),
                noise=noise,
                clip_denoised=args.clip_denoised,
                device=dist_util.dev(),
                progress=args.progress,
                labels=labels,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_runtime = time.perf_counter() - started
        total_sampling_seconds += batch_runtime
        runtime_per_sample = batch_runtime / current_batch

        rows = []
        for batch_index in range(current_batch):
            index = completed + batch_index + 1
            raw = sample[batch_index, 0].detach().float().cpu().numpy()
            binary = (raw > args.sample_threshold).astype(np.uint8) * 255
            generated_porosity = float((binary > 0).mean())
            binary_path = output_dir / f"sample_{index:04d}_binary.tif"
            tifffile.imwrite(binary_path, binary)
            raw_path = ""
            if args.save_raw:
                raw_file = output_dir / f"sample_{index:04d}_raw.tif"
                tifffile.imwrite(raw_file, raw.astype(np.float32))
                raw_path = str(raw_file)
            rows.append(
                {
                    "sample_index": index,
                    "seed": batch_seed,
                    "seed_scope": (
                        "sample" if current_batch == 1 else "sampling_batch"
                    ),
                    "target_porosity": args.target_porosity,
                    "generated_porosity": generated_porosity,
                    "absolute_porosity_error": (
                        abs(generated_porosity - args.target_porosity)
                    ),
                    "sampler": sampler_name,
                    "runtime_seconds": runtime_per_sample,
                    "binary_path": str(binary_path),
                    "raw_path": raw_path,
                }
            )
        if dist_util.get_rank() == 0:
            with log_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=_sample_fields()).writerows(rows)
        completed += current_batch

    dist_util.barrier()
    if dist_util.get_rank() == 0:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        compute = {
            "parameter_count": parameter_count,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "sample_count": args.num_samples,
            "wall_clock_sampling_time_seconds": total_sampling_seconds,
            "seconds_per_sample": total_sampling_seconds / args.num_samples,
            "seconds_per_reverse_step": total_sampling_seconds
            / (args.num_samples * diffusion.num_timesteps),
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
                if torch.cuda.is_available()
                else None
            ),
            "gpu_model": (
                torch.cuda.get_device_name(dist_util.dev())
                if torch.cuda.is_available()
                else None
            ),
            "volume_shape": list(shape),
            "diffusion_steps": diffusion.num_timesteps,
            "sampler": sampler_name,
            "model_path": str(checkpoint),
        }
        (output_dir / "compute_sample.json").write_text(
            json.dumps(compute, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        experiment_variant="A_full",
        model_path="",
        output_dir="revision_outputs/full_model_samples",
        data_dir="",
        seed=1001,
        deterministic=False,
        target_porosity=0.18,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        clip_denoised=True,
        progress=False,
        save_raw=True,
        sample_threshold=0.0,
        binary_threshold=None,
        pore_is_high=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser(
        description="Sample from an explicit checkpoint and log porosity control."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
