import argparse
import math

from .demo_sinddpm import UNetModel
from . import demo_gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        # Retain the submitted implementation's learned-variance behavior.
        # The code now exposes and logs MSE, VB, and their weighted total so
        # the revised manuscript can describe the objective truthfully.
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",   # cosine or linear
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        vb_weight=1.0,
    )

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=64,
        num_res_blocks=1,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=16,
        attention_resolutions="2",
        channel_mult="1,2,4",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        use_condition=True,
        condition_dim=1,
    )
    res.update(diffusion_defaults())
    return res

def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    use_condition,
    condition_dim,
    vb_weight,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        use_condition=use_condition,
        condition_dim=condition_dim,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        vb_weight=vb_weight,
    )
    return model, diffusion

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=True,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    use_condition=True,
    condition_dim=1,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=1,
        model_channels=num_channels,
        out_channels=2 if learn_sigma else 1,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        use_condition=use_condition,
        condition_dim=condition_dim,
    )

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=True,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    vb_weight=1.0,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        vb_weight=vb_weight,
    )

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def adjust_scales2image(real_, opt):
    """Populate legacy scale attributes without interpolating phase labels.

    The original helper converted a binary 3D tensor to uint8, applied cubic
    interpolation, and forced a CUDA tensor.  Besides preventing CPU use, that
    introduced non-binary phase values.  The loader now performs any requested
    resizing with nearest-neighbour interpolation; this helper only computes
    the scale metadata retained by the command-line interface.
    """

    if len(real_.shape) < 3:
        raise ValueError(f"expected spatial dimensions, received {real_.shape}")
    spatial = tuple(int(value) for value in real_.shape[-3:])
    min_dim, max_dim = min(spatial), max(spatial)
    if opt.scale_factor_init <= 0 or opt.scale_factor_init >= 1:
        raise ValueError("scale_factor_init must be between 0 and 1")
    if opt.min_size <= 0 or opt.max_size <= 0:
        raise ValueError("min_size and max_size must be positive")

    opt.scale1 = min(float(opt.max_size) / max_dim, 1.0)
    scaled_min = max(1.0, min_dim * opt.scale1)
    if scaled_min <= opt.min_size:
        opt.stop_scale = 0
        opt.num_scales = 1
        opt.scale_factor = 1.0
    else:
        opt.stop_scale = max(
            1,
            int(math.ceil(math.log(opt.min_size / scaled_min, opt.scale_factor_init))),
        )
        opt.num_scales = opt.stop_scale + 1
        opt.scale_factor = math.pow(opt.min_size / scaled_min, 1.0 / opt.stop_scale)
    return real_
