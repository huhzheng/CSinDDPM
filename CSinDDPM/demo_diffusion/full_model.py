"""Guard the full-model distribution against unconditional runs."""


def require_porosity_condition(use_condition):
    if not use_condition:
        raise ValueError(
            "This full-model distribution requires use_condition=true: "
            "porosity and Conditional ResBlocks cannot be disabled here."
        )
