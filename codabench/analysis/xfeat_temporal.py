"""Extra temporal feature blocks for the harness (see xfeat.py for the interface;
sprint 2026-09-29, brief prompts/2026-09-29_temporal_spatial_features.md)."""

import numpy as np  # noqa: F401

from xfeat import FB4, Block, band_pass, covs, lag_samples, register, tangent_space  # noqa: F401
