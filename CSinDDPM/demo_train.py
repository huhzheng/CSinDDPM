"""Train CSinDDPM on one explicitly encoded 3D binary volume."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from demo_diffusion import dist_util, logger
from demo_diffusion.demo_image_datasets import compute_porosity, load_data, read_binary_tiff
from demo_diffusion.demo_script_util import (
    add_dict_to_argparser,
    adjust_scales2image,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from demo_diffusion.demo_train_util import TrainLoop
from demo_diffusion.full_model import require_porosity_condition
from demo_diffusion.reproducibility import (
    seed_everything,
    write_model_summary,
    write_run_metadata,
)
from demo_diffusion.resample import create_named_schedule_sampler


def main():
    args = create_argparser().parse_args()
    require_porosity_condition(args.use_condition)
    data_path = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"--data_dir does not identify a TIFF: {data_path}")

    dist_util.setup_dist()
    seed_everything(
        args.seed + dist_util.get_rank(), deterministic=args.deterministic
    )
    logger.configure(dir=str(output_dir))

    encoded = read_binary_tiff(
        data_path,
        threshold=args.binary_threshold,
        pore_is_high=args.pore_is_high,
    )
    real = torch.from_numpy(encoded).unsqueeze(0).unsqueeze(0)
    adjust_scales2image(real, args)
    source_porosity = float(compute_porosity(real).item())
    logger.log(
        "source volume: "
        f"shape={tuple(encoded.shape)}, pore=+1, grain=-1, "
        f"porosity={source_porosity:.8f}"
    )

    validation_real = None
    validation_path = None
    validation_source_role = None
    if args.select_best_checkpoint:
        validation_path = (
            Path(args.validation_data_dir).expanduser().resolve()
            if args.validation_data_dir
            else data_path
        )
        if not validation_path.is_file():
            raise FileNotFoundError(
                f"--validation_data_dir does not identify a TIFF: {validation_path}"
            )
        validation_encoded = read_binary_tiff(
            validation_path,
            threshold=args.binary_threshold,
            pore_is_high=args.pore_is_high,
        )
        if validation_encoded.shape != encoded.shape:
            raise ValueError(
                "validation volume shape must match the training volume: "
                f"{validation_encoded.shape} != {encoded.shape}"
            )
        validation_real = (
            torch.from_numpy(validation_encoded).unsqueeze(0).unsqueeze(0)
        )
        validation_source_role = (
            "training_volume_fixed_noise_proxy"
            if validation_path == data_path
            else "independent_validation_volume"
        )
        logger.log(
            "best-checkpoint selection enabled: "
            f"source={validation_path}, role={validation_source_role}, "
            f"metric={args.best_checkpoint_metric}, "
            f"trials={args.validation_trials}, seed={args.validation_seed}"
        )

    if dist_util.get_rank() == 0:
        run_config = dict(vars(args))
        run_config.update(
            resolved_data_dir=str(data_path),
            resolved_output_dir=str(output_dir),
            source_shape=list(encoded.shape),
            source_porosity=source_porosity,
            phase_convention={"pore": 1.0, "grain": -1.0},
            world_size=dist_util.get_world_size(),
            validation_source=(str(validation_path) if validation_path else None),
            validation_source_role=validation_source_role,
        )
        write_run_metadata(
            output_dir,
            run_config,
            run_type="train",
            repository_root=Path(__file__).resolve().parent,
        )

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if dist_util.get_rank() == 0:
        architecture = write_model_summary(
            output_dir, model, requested_attention=args.attention_resolutions
        )
        if architecture["attention_block_count"] == 0:
            logger.warn(
                "submitted backbone contains no active AttentionBlock; "
                "--attention_resolutions is retained for compatibility only"
            )
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion
    )

    logger.log("creating deterministic single-volume loader...")
    data = load_data(
        data_dir=str(data_path),
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        scale_init=args.scale1,
        scale_factor=args.scale_factor,
        stop_scale=args.stop_scale,
        current_scale=args.stop_scale,
        binary_threshold=args.binary_threshold,
        pore_is_high=args.pore_is_high,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        output_dir=output_dir,
        train_steps=args.train_steps,
        dirty_probability=args.dirty_probability,
        dirty_min_frac=args.dirty_min_frac,
        dirty_max_frac=args.dirty_max_frac,
        debug_save_augmentation=args.debug_save_augmentation,
        seed=args.seed,
        use_condition=args.use_condition,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        validation_data=validation_real,
        validation_source=(str(validation_path) if validation_path else None),
        validation_source_role=validation_source_role,
        validation_trials=args.validation_trials,
        validation_seed=args.validation_seed,
        best_checkpoint_metric=args.best_checkpoint_metric,
        validate_ema=args.validate_ema,
    ).run_loop()


def create_argparser():
    defaults = dict(
        experiment_variant="A_full",
        data_dir="",
        output_dir="revision_outputs/full_model_train",
        schedule_sampler="uniform",
        seed=2026,
        deterministic=False,
        lr=1e-4,
        weight_decay=0.0,
        train_steps=60000,
        dirty_probability=0.30,
        dirty_min_frac=0.10,
        dirty_max_frac=0.30,
        debug_save_augmentation=False,
        binary_threshold=None,
        pore_is_high=True,
        num_workers=0,
        scale_factor_init=0.75,
        min_size=16,
        max_size=64,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=200,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        select_best_checkpoint=True,
        validation_data_dir="",
        validation_trials=16,
        validation_seed=314159,
        best_checkpoint_metric="total_loss",
        validate_ema=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser(
        description=(
            "Train CSinDDPM. Binary TIFF values above the threshold are "
            "treated as pore and encoded as +1 by default."
        )
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
