"""Condition and objective propagation tests for DDPM and DDIM."""

import numpy as np
import pytest
import torch
from torch import nn

from demo_diffusion import demo_gaussian_diffusion as gd
from demo_diffusion.demo_sinddpm import ResBlock, UNetModel


class RecordingModel(nn.Module):
    def __init__(self, learned_variance=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.learned_variance = learned_variance
        self.labels_seen = []

    def forward(self, x, timesteps, labels=None):
        self.labels_seen.append(None if labels is None else labels.detach().clone())
        channels = x.shape[1] * (2 if self.learned_variance else 1)
        return torch.zeros((x.shape[0], channels, *x.shape[2:]), device=x.device)


def make_diffusion(var_type=gd.ModelVarType.FIXED_SMALL, vb_weight=1.0):
    return gd.GaussianDiffusion(
        betas=np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64),
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=var_type,
        loss_type=gd.LossType.MSE,
        vb_weight=vb_weight,
    )


@pytest.mark.parametrize("sampler_name", ["p_sample_loop", "ddim_sample_loop"])
def test_condition_is_forwarded_at_every_sampling_step(sampler_name):
    diffusion = make_diffusion()
    model = RecordingModel()
    labels = torch.tensor([[0.18], [0.20]])
    noise = torch.zeros((2, 1, 4, 4, 4))
    sample = getattr(diffusion, sampler_name)(
        model,
        noise.shape,
        noise=noise,
        device=torch.device("cpu"),
        labels=labels,
    )
    assert sample.shape == noise.shape
    assert len(model.labels_seen) == diffusion.num_timesteps
    assert all(torch.equal(seen, labels) for seen in model.labels_seen)


def test_fixed_variance_loss_is_explicit_pure_mse():
    diffusion = make_diffusion()
    model = RecordingModel()
    x = torch.ones((2, 1, 4, 4, 4))
    t = torch.tensor([0, 2])
    terms = diffusion.training_losses(model, x, t, labels=torch.ones((2, 1)))
    assert {"mse", "vb", "total_loss", "loss"}.issubset(terms)
    assert torch.equal(terms["vb"], torch.zeros_like(terms["vb"]))
    assert torch.allclose(terms["total_loss"], terms["mse"])


def test_learned_variance_loss_reports_weighted_vb_component():
    diffusion = make_diffusion(gd.ModelVarType.LEARNED_RANGE, vb_weight=0.25)
    model = RecordingModel(learned_variance=True)
    x = torch.ones((2, 1, 4, 4, 4))
    t = torch.tensor([1, 3])
    terms = diffusion.training_losses(model, x, t, labels=torch.ones((2, 1)))
    assert torch.allclose(
        terms["total_loss"], terms["mse"] + 0.25 * terms["vb"]
    )


def test_full_model_rejects_unconditional_backbone():
    with pytest.raises(ValueError, match="requires use_condition=true"):
        UNetModel(
            image_size=8,
            in_channels=1,
            model_channels=32,
            out_channels=1,
            num_res_blocks=1,
            attention_resolutions=(),
            channel_mult=(1,),
            use_condition=False,
        )


def test_conditioned_backbone_retains_submitted_checkpoint_keys():
    model = UNetModel(
        image_size=8,
        in_channels=1,
        model_channels=32,
        out_channels=1,
        num_res_blocks=1,
        attention_resolutions=(),
        channel_mult=(1,),
        use_condition=True,
    )
    keys = set(model.state_dict())
    assert "cond_embedding2.condEmbedding2.0.weight" in keys
    assert "input_blocks.1.0.cond_layers.1.weight" in keys
    resblocks = [module for module in model.modules() if isinstance(module, ResBlock)]
    assert resblocks
    assert all(module.cond_layers is not None for module in resblocks)
    with pytest.raises(ValueError, match="no condition"):
        model(torch.randn((1, 1, 8, 8, 8)), torch.tensor([1]))


def test_conditional_gradient_checkpointing_propagates_condition_gradient():
    model = UNetModel(
        image_size=8,
        in_channels=1,
        model_channels=32,
        out_channels=1,
        num_res_blocks=1,
        attention_resolutions=(),
        channel_mult=(1,),
        use_condition=True,
        use_checkpoint=True,
    )
    # Nonzero output weights exercise the conditional gradient path.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("out_layers.3.weight") or name == "out.2.weight":
                parameter.normal_(0, 0.01)
    labels = torch.tensor([[0.18]], requires_grad=True)
    output = model(
        torch.randn((1, 1, 8, 8, 8), requires_grad=True),
        torch.tensor([1]),
        labels=labels,
    )
    output.sum().backward()
    assert labels.grad is not None
    assert torch.isfinite(labels.grad).all()
    assert labels.grad.abs().sum() > 0


def test_scale_shift_condition_path_has_valid_shape():
    model = UNetModel(
        image_size=8,
        in_channels=1,
        model_channels=32,
        out_channels=1,
        num_res_blocks=1,
        attention_resolutions=(),
        channel_mult=(1,),
        use_condition=True,
        use_scale_shift_norm=True,
    )
    with torch.no_grad():
        output = model(
            torch.randn((1, 1, 8, 8, 8)),
            torch.tensor([1]),
            labels=torch.tensor([[0.18]]),
        )
    assert output.shape == (1, 1, 8, 8, 8)


def test_kl_objective_still_exposes_mse_vb_and_total_keys():
    diffusion = gd.GaussianDiffusion(
        betas=np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64),
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.KL,
        vb_weight=0.5,
    )
    model = RecordingModel()
    terms = diffusion.training_losses(
        model,
        torch.ones((2, 1, 4, 4, 4)),
        torch.tensor([0, 2]),
        labels=torch.ones((2, 1)),
    )
    assert torch.equal(terms["mse"], torch.zeros_like(terms["mse"]))
    assert torch.allclose(terms["total_loss"], 0.5 * terms["vb"])
