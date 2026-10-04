"""Analyze supplied permeability values; this module is not a flow solver."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def analyze_permeability_table(csv_path) -> dict:
    """Relate measured/simulated permeability to porosity and connectivity.

    The CSV must contain ``porosity`` and at least one of ``permeability_x``,
    ``permeability_y``, or ``permeability_z``. Optional
    ``spanning_porosity_x/y/z`` columns permit the reviewer-requested joint
    interpretation k=f(phi, connectivity). No permeability is inferred from
    morphology alone.
    """

    table = pd.read_csv(csv_path)
    if "porosity" not in table:
        raise ValueError("CSV must contain a porosity column")
    output = {"row_count": int(len(table)), "directions": {}}
    for direction in "xyz":
        k_column = f"permeability_{direction}"
        c_column = f"spanning_porosity_{direction}"
        if k_column not in table:
            continue
        columns = ["porosity", k_column] + ([c_column] if c_column in table else [])
        subset = table[columns].dropna()
        subset = subset[subset[k_column] > 0]
        if len(subset) < 3:
            output["directions"][direction] = {
                "status": "PENDING_AT_LEAST_3_POSITIVE_ROWS",
                "n": int(len(subset)),
            }
            continue
        log_k = np.log10(subset[k_column].to_numpy(dtype=float))
        phi = subset["porosity"].to_numpy(dtype=float)
        rho_phi, p_phi = stats.spearmanr(phi, log_k)
        predictors = [np.ones(len(subset)), phi]
        names = ["intercept", "porosity"]
        if c_column in subset:
            predictors.append(subset[c_column].to_numpy(dtype=float))
            names.append(c_column)
        design = np.column_stack(predictors)
        coefficients, _, _, _ = np.linalg.lstsq(design, log_k, rcond=None)
        prediction = design @ coefficients
        ss_res = float(np.square(log_k - prediction).sum())
        ss_tot = float(np.square(log_k - log_k.mean()).sum())
        row = {
            "status": "analyzed_supplied_values",
            "n": int(len(subset)),
            "spearman_porosity_log10k": _finite_or_none(rho_phi),
            "spearman_porosity_p": _finite_or_none(p_phi),
            "ols_log10k_coefficients": dict(zip(names, coefficients.tolist())),
            "ols_r_squared": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
        }
        if c_column in subset:
            rho_c, p_c = stats.spearmanr(subset[c_column], log_k)
            row["spearman_connectivity_log10k"] = _finite_or_none(rho_c)
            row["spearman_connectivity_p"] = _finite_or_none(p_c)
        output["directions"][direction] = row
    if not output["directions"]:
        raise ValueError("no permeability_x/y/z columns were found")
    return output


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None
