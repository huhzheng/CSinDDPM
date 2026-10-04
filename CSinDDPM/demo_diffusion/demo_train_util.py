"""Training loop with auditable augmentation, losses, and checkpoints."""

from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import tifffile
import torch as th
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .augmentation import apply_dirty_cuboid
from .demo_image_datasets import compute_porosity
from .fp16_util import MixedPrecisionTrainer
from .full_model import require_porosity_condition
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler


class TrainLoop:
    """Train until an explicit total step count is reached."""

    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        output_dir,
        train_steps,
        dirty_probability=0.30,
        dirty_min_frac=0.10,
        dirty_max_frac=0.30,
        debug_save_augmentation=False,
        seed=0,
        use_condition=True,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        validation_data=None,
        validation_source=None,
        validation_source_role=None,
        validation_trials=16,
        validation_seed=314159,
        best_checkpoint_metric="total_loss",
        validate_ema=True,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = int(batch_size)
        self.microbatch = microbatch if microbatch > 0 else self.batch_size
        self.lr = float(lr)
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(value) for value in str(ema_rate).split(",")]
        )
        self.log_interval = int(log_interval)
        self.save_interval = int(save_interval)
        self.resume_checkpoint = resume_checkpoint
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_steps = int(train_steps)
        self.dirty_probability = float(dirty_probability)
        self.dirty_min_frac = float(dirty_min_frac)
        self.dirty_max_frac = float(dirty_max_frac)
        self.debug_save_augmentation = bool(debug_save_augmentation)
        self.seed = int(seed)
        require_porosity_condition(use_condition)
        self.use_condition = True
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = float(weight_decay)
        self.validation_data = (
            validation_data.detach().clone().float().cpu()
            if validation_data is not None
            else None
        )
        self.validation_source = (
            str(validation_source) if validation_source is not None else None
        )
        self.validation_source_role = validation_source_role
        self.validation_trials = int(validation_trials)
        self.validation_seed = int(validation_seed)
        self.best_checkpoint_metric = str(best_checkpoint_metric)
        self.validate_ema = bool(validate_ema)

        if self.train_steps <= 0:
            raise ValueError("train_steps must be positive")
        if self.save_interval <= 0 or self.log_interval <= 0:
            raise ValueError("save_interval and log_interval must be positive")
        if self.validation_data is not None:
            if self.validation_trials <= 0:
                raise ValueError("validation_trials must be positive")
            if self.best_checkpoint_metric not in {"mse", "vb", "total_loss"}:
                raise ValueError(
                    "best_checkpoint_metric must be one of: mse, vb, total_loss"
                )
            if self.validation_data.ndim != 5 or self.validation_data.shape[:2] != (
                1,
                1,
            ):
                raise ValueError(
                    "validation_data must have shape [1, 1, Z, Y, X], received "
                    f"{tuple(self.validation_data.shape)}"
                )

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist_util.get_world_size()
        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.opt = AdamW(
            self.mp_trainer.master_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params) for _ in self.ema_rate
            ]

        if dist_util.get_world_size() > 1:
            device = dist_util.dev()
            device_ids = [device.index] if device.type == "cuda" else None
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=device_ids,
                output_device=device.index if device.type == "cuda" else None,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.use_ddp = False
            self.ddp_model = self.model

        self.train_log_path = self.output_dir / "train_log.csv"
        if dist_util.get_rank() == 0 and not self.train_log_path.exists():
            with self.train_log_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self._training_fields())
                writer.writeheader()

        self.validation_log_path = self.output_dir / "validation_log.csv"
        self.best_manifest_path = self.output_dir / "best_checkpoints.json"
        self._selection_candidates = []
        self._best_by_variant = {}
        self._best_overall = None
        if self.validation_data is not None:
            self._initialize_selection_state()

    @property
    def global_step(self):
        return self.step + self.resume_step

    @staticmethod
    def _training_fields():
        return [
            "step",
            "mse",
            "vb",
            "total_loss",
            "weighted_total_loss",
            "dirty_applied",
            "dirty_mode",
            "cuboid_bbox",
            "porosity_before",
            "porosity_after",
            "learning_rate",
        ]

    @staticmethod
    def _validation_fields():
        return [
            "step",
            "variant",
            "ema_rate",
            "metric",
            "score",
            "mean_mse",
            "mean_vb",
            "mean_total_loss",
            "source_checkpoint",
            "is_new_variant_best",
            "is_new_overall_best",
        ]

    def _fixed_validation_timesteps(self):
        count = self.validation_trials
        total = int(self.diffusion.num_timesteps)
        return [
            min(total - 1, int((trial + 0.5) * total / count))
            for trial in range(count)
        ]

    def _selection_protocol(self):
        source_hash = None
        if self.validation_source and Path(self.validation_source).is_file():
            source_hash = _sha256_file(Path(self.validation_source))
        return {
            "schema_version": 1,
            "direction": "minimize",
            "metric": f"mean_{self.best_checkpoint_metric}",
            "evaluated_at_every_saved_checkpoint": True,
            "validation_augmentation": "disabled",
            "validation_source": self.validation_source,
            "validation_source_sha256": source_hash,
            "validation_source_role": self.validation_source_role,
            "validation_seed": self.validation_seed,
            "validation_trials": self.validation_trials,
            "fixed_timesteps": self._fixed_validation_timesteps(),
            "ordinary_and_ema_ranked_separately": True,
            "include_ema_in_overall_ranking": self.validate_ema,
            "tie_break": "lower step, then ordinary model before EMA",
        }

    def _initialize_selection_state(self):
        protocol = self._selection_protocol()
        if self.best_manifest_path.exists():
            payload = json.loads(
                self.best_manifest_path.read_text(encoding="utf-8")
            )
            if payload.get("selection_protocol") != protocol:
                raise RuntimeError(
                    "existing best_checkpoints.json uses a different selection "
                    "protocol; resume into the original directory only with the "
                    "same validation arguments"
                )
            self._selection_candidates = list(payload.get("candidates", []))
            self._best_by_variant = dict(payload.get("best_by_variant", {}))
            self._best_overall = payload.get("best_overall")
        elif self.validation_log_path.exists():
            raise RuntimeError(
                "validation_log.csv exists without best_checkpoints.json; preserve "
                "the partial directory and restart in a fresh output directory"
            )

        if dist_util.get_rank() == 0 and not self.validation_log_path.exists():
            with self.validation_log_path.open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                csv.DictWriter(
                    stream, fieldnames=self._validation_fields()
                ).writeheader()
        if dist_util.get_rank() == 0:
            self._write_selection_manifest(status="running")

    def _write_selection_manifest(self, status):
        if dist_util.get_rank() != 0:
            return
        role_notice = (
            "Selection uses a fixed-noise proxy on the training volume. It is "
            "reproducible but is not independent generalization or morphology "
            "evidence."
            if self.validation_source_role == "training_volume_fixed_noise_proxy"
            else "Selection uses the explicitly supplied validation volume."
        )
        payload = {
            "schema_version": 1,
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_step": self.global_step,
            "selection_protocol": self._selection_protocol(),
            "candidates": self._selection_candidates,
            "best_by_variant": self._best_by_variant,
            "best_overall": self._best_overall,
            "interpretation_boundary": role_notice,
        }
        _atomic_write_json(self.best_manifest_path, payload)

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist_util.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint,
                    map_location=dist_util.dev(),
                )
            )
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist_util.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint,
                map_location=dist_util.dev(),
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = (
            Path(main_checkpoint).parent / f"opt{self.resume_step:06d}.pt"
        )
        if opt_checkpoint.exists():
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                str(opt_checkpoint),
                map_location=dist_util.dev(),
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        if self.resume_step > self.train_steps:
            raise ValueError(
                f"checkpoint step {self.resume_step} is above train_steps "
                f"{self.train_steps}"
            )
        if self.validation_data is not None and self.resume_step:
            expected_variants = {"model"}
            if self.validate_ema:
                expected_variants.update(
                    f"ema_{rate}" for rate in self.ema_rate
                )
            completed_variants = {
                row["variant"]
                for row in self._selection_candidates
                if int(row["step"]) == self.resume_step
            }
            if not expected_variants.issubset(completed_variants):
                dist_util.barrier()
                if dist_util.get_rank() == 0:
                    self._validate_and_update_best(self.resume_step)
                dist_util.barrier()
                dist_util.sync_params(self.model.parameters())
        if self.resume_step == self.train_steps:
            if dist_util.get_rank() == 0:
                logger.log(
                    "resume checkpoint already equals train_steps; no optimizer "
                    "steps are required"
                )
                if self.validation_data is not None:
                    self._write_selection_manifest(status="complete")
            dist_util.barrier()
            return
        if th.cuda.is_available():
            th.cuda.reset_peak_memory_stats()
            th.cuda.synchronize()
        started = time.perf_counter()
        while self.global_step < self.train_steps:
            batch, _ = next(self.data)
            rows = self.run_step(batch)
            self.step += 1
            self.log_step()
            self._write_rows(rows)
            if self.global_step % self.log_interval == 0:
                logger.dumpkvs()
            if self.global_step % self.save_interval == 0:
                self.save()
        if self.global_step % self.save_interval != 0:
            self.save()
        logger.dumpkvs()
        if th.cuda.is_available():
            th.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if dist_util.get_rank() == 0:
            parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
            trainable_count = sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
            compute = {
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_count,
                "measured_steps": self.step,
                "wall_clock_training_time_seconds": elapsed,
                "seconds_per_step": elapsed / max(1, self.step),
                "peak_gpu_memory_mb": (
                    th.cuda.max_memory_allocated() / (1024 ** 2)
                    if th.cuda.is_available()
                    else None
                ),
                "gpu_model": (
                    th.cuda.get_device_name(dist_util.dev())
                    if th.cuda.is_available()
                    else None
                ),
            }
            (self.output_dir / "compute_train.json").write_text(
                json.dumps(compute, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if self.validation_data is not None:
                self._write_selection_manifest(status="complete")
        dist_util.barrier()

    def run_step(self, batch):
        rows = self.forward_backward(batch)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        return rows

    def forward_backward(self, batch):
        self.mp_trainer.zero_grad()
        rows = []
        for offset in range(0, batch.shape[0], self.microbatch):
            micro = batch[offset : offset + self.microbatch].to(dist_util.dev())
            last_batch = (offset + self.microbatch) >= batch.shape[0]
            generator = th.Generator(device="cpu")
            generator.manual_seed(
                self.seed
                + self.global_step * 1009
                + offset
                + dist_util.get_rank() * 1_000_003
            )
            micro, metadata = apply_dirty_cuboid(
                micro,
                probability=self.dirty_probability,
                min_frac=self.dirty_min_frac,
                max_frac=self.dirty_max_frac,
                generator=generator,
            )
            labels = compute_porosity(micro).reshape(micro.shape[0], -1)
            t, weights = self.schedule_sampler.sample(
                micro.shape[0], dist_util.dev()
            )
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                labels=labels,
            )
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["total_loss"].detach()
                )
            weighted_loss = (losses["total_loss"] * weights).mean()
            log_loss_dict(
                self.diffusion,
                t,
                {key: value * weights for key, value in losses.items()},
            )
            self.mp_trainer.backward(weighted_loss)
            if self.debug_save_augmentation and dist_util.get_rank() == 0:
                self._save_debug_volume(micro, offset)
            rows.append(
                {
                    "step": self.global_step + 1,
                    "mse": float(losses["mse"].mean().detach().cpu()),
                    "vb": float(losses["vb"].mean().detach().cpu()),
                    "total_loss": float(
                        losses["total_loss"].mean().detach().cpu()
                    ),
                    "weighted_total_loss": float(weighted_loss.detach().cpu()),
                    "dirty_applied": bool(metadata["dirty_applied"]),
                    "dirty_mode": metadata["dirty_mode"],
                    "cuboid_bbox": json.dumps(metadata["cuboid_bbox"]),
                    "porosity_before": metadata["porosity_before"],
                    "porosity_after": metadata["porosity_after"],
                    "learning_rate": self.opt.param_groups[0]["lr"],
                }
            )
        return rows

    def _save_debug_volume(self, micro, offset):
        debug_dir = self.output_dir / "debug_augmentation"
        debug_dir.mkdir(parents=True, exist_ok=True)
        array = (micro[0, 0].detach().cpu().numpy() > 0).astype(np.uint8) * 255
        tifffile.imwrite(
            debug_dir / f"step_{self.global_step + 1:06d}_micro_{offset:03d}.tif",
            array,
        )

    def _write_rows(self, rows):
        if dist_util.get_rank() != 0:
            return
        with self.train_log_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self._training_fields())
            writer.writerows(rows)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def log_step(self):
        logger.logkv("step", self.global_step)
        logger.logkv("samples", self.global_step * self.global_batch)

    def _evaluate_current_weights(self):
        device = dist_util.dev()
        x_start = self.validation_data.to(device)
        labels = compute_porosity(x_start).reshape(x_start.shape[0], -1)
        totals = {"mse": [], "vb": [], "total_loss": []}
        timesteps = self._fixed_validation_timesteps()
        was_training = self.model.training
        self.model.eval()
        try:
            with th.no_grad():
                for trial, timestep in enumerate(timesteps):
                    generator = th.Generator(device="cpu")
                    generator.manual_seed(self.validation_seed + trial)
                    noise = th.randn(
                        tuple(x_start.shape),
                        generator=generator,
                        dtype=x_start.dtype,
                        device="cpu",
                    ).to(device)
                    t = th.tensor([timestep], device=device, dtype=th.long)
                    losses = self.diffusion.training_losses(
                        self.model,
                        x_start,
                        t,
                        labels=labels,
                        noise=noise,
                    )
                    for key in totals:
                        totals[key].append(
                            float(losses[key].mean().detach().float().cpu())
                        )
        finally:
            self.model.train(was_training)
        metrics = {key: float(np.mean(values)) for key, values in totals.items()}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"non-finite validation metrics: {metrics}")
        return metrics

    @staticmethod
    def _candidate_is_better(candidate, current):
        if current is None:
            return True
        candidate_key = (
            float(candidate["score"]),
            int(candidate["step"]),
            candidate["variant"] != "model",
            candidate["variant"],
        )
        current_key = (
            float(current["score"]),
            int(current["step"]),
            current["variant"] != "model",
            current["variant"],
        )
        return candidate_key < current_key

    def _promote_checkpoint(self, candidate, alias_name):
        source = self.output_dir / candidate["source_checkpoint"]
        if not source.is_file():
            raise FileNotFoundError(f"candidate checkpoint is missing: {source}")
        alias = self.output_dir / alias_name
        _atomic_link_or_copy(source, alias)
        digest = _sha256_file(source)
        promoted = dict(candidate)
        promoted.update(
            alias=alias.name,
            source_checkpoint_sha256=digest,
            alias_sha256=digest,
        )
        return promoted

    def _record_validation_candidate(self, candidate):
        variant = candidate["variant"]
        previous_variant_best = self._best_by_variant.get(variant)
        new_variant_best = self._candidate_is_better(
            candidate, previous_variant_best
        )
        new_overall_best = self._candidate_is_better(
            candidate, self._best_overall
        )

        if new_variant_best:
            alias = (
                "best_model.pt"
                if variant == "model"
                else f"best_{variant}.pt"
            )
            self._best_by_variant[variant] = self._promote_checkpoint(
                candidate, alias
            )
        if new_overall_best:
            self._best_overall = self._promote_checkpoint(
                candidate, "best_overall.pt"
            )
        return new_variant_best, new_overall_best

    def _validate_and_update_best(self, step):
        if self.validation_data is None or dist_util.get_rank() != 0:
            return
        ordinary_checkpoint = self.output_dir / f"model{step:06d}.pt"
        if not ordinary_checkpoint.is_file():
            raise FileNotFoundError(
                f"cannot validate missing checkpoint: {ordinary_checkpoint}"
            )

        existing = {
            (int(row["step"]), row["variant"])
            for row in self._selection_candidates
        }
        candidate_specs = [("model", None, ordinary_checkpoint.name, None)]
        if self.validate_ema:
            candidate_specs.extend(
                (
                    f"ema_{rate}",
                    rate,
                    f"ema_{rate}_{step:06d}.pt",
                    params,
                )
                for rate, params in zip(self.ema_rate, self.ema_params)
            )

        ordinary_state = None
        candidates = []
        try:
            for variant, rate, filename, params in candidate_specs:
                if (step, variant) in existing:
                    continue
                if variant != "model":
                    if ordinary_state is None:
                        ordinary_state = dist_util.load_state_dict(
                            str(ordinary_checkpoint), map_location="cpu"
                        )
                    ema_checkpoint = self.output_dir / filename
                    if not ema_checkpoint.is_file():
                        raise FileNotFoundError(
                            f"cannot validate missing EMA checkpoint: {ema_checkpoint}"
                        )
                    ema_state = self.mp_trainer.master_params_to_state_dict(params)
                    self.model.load_state_dict(ema_state)
                metrics = self._evaluate_current_weights()
                candidate = {
                    "step": int(step),
                    "variant": variant,
                    "ema_rate": rate,
                    "metric": f"mean_{self.best_checkpoint_metric}",
                    "score": metrics[self.best_checkpoint_metric],
                    "mean_mse": metrics["mse"],
                    "mean_vb": metrics["vb"],
                    "mean_total_loss": metrics["total_loss"],
                    "source_checkpoint": filename,
                }
                candidates.append(candidate)
        finally:
            if ordinary_state is not None:
                self.model.load_state_dict(ordinary_state)

        rows = []
        for candidate in candidates:
            self._selection_candidates.append(candidate)
            new_variant_best, new_overall_best = self._record_validation_candidate(
                candidate
            )
            row = dict(candidate)
            row.update(
                is_new_variant_best=new_variant_best,
                is_new_overall_best=new_overall_best,
            )
            rows.append(row)
            logger.log(
                "checkpoint validation: "
                f"step={step}, variant={candidate['variant']}, "
                f"{candidate['metric']}={candidate['score']:.8g}, "
                f"new_variant_best={new_variant_best}, "
                f"new_overall_best={new_overall_best}"
            )
        if rows:
            with self.validation_log_path.open(
                "a", newline="", encoding="utf-8"
            ) as stream:
                csv.DictWriter(
                    stream, fieldnames=self._validation_fields()
                ).writerows(rows)
            self._write_selection_manifest(status="running")

    def save(self):
        step = self.global_step

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist_util.get_rank() != 0:
                return
            filename = (
                f"model{step:06d}.pt"
                if not rate
                else f"ema_{rate}_{step:06d}.pt"
            )
            with (self.output_dir / filename).open("wb") as stream:
                th.save(state_dict, stream)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        if dist_util.get_rank() == 0:
            with (self.output_dir / f"opt{step:06d}.pt").open("wb") as stream:
                th.save(self.opt.state_dict(), stream)
        dist_util.barrier()
        if self.validation_data is not None and dist_util.get_rank() == 0:
            self._validate_and_update_best(step)
        dist_util.barrier()
        if self.validation_data is not None:
            dist_util.sync_params(self.model.parameters())


def _temporary_sibling(path):
    return path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )


def _atomic_write_json(path, payload):
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_link_or_copy(source, destination):
    temporary = _temporary_sibling(destination)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_resume_step_from_filename(filename):
    split = str(filename).split("model")
    if len(split) < 2:
        return 0
    try:
        return int(split[-1].split(".")[0])
    except ValueError:
        return 0


def get_blob_logdir():
    """Compatibility helper retained for downstream imports."""

    return logger.get_dir()


def find_resume_checkpoint():
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    path = Path(main_checkpoint).parent / f"ema_{rate}_{step:06d}.pt"
    return str(path) if path.exists() else None


def log_loss_dict(diffusion, timesteps, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for timestep, value in zip(
            timesteps.cpu().numpy(), values.detach().cpu().numpy()
        ):
            quartile = int(4 * timestep / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", value)
