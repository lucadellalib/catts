# ==============================================================================
# Copyright 2026 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""PyTorch Hub entry point."""

import os
from typing import Any

import torch

from catts import CATTS

# Make sure it is consistent with requirements.txt and README.md
dependencies = [
    "huggingface_hub",
    "numpy",
    "safetensors",
    "torch",
    "transformers",
]


def catts(
    release: "str" = "v0.0.1",
    checkpoint_name: "str" = "catts.pt",
    map_location: "str" = "cpu",
    **kwargs: "Any",
) -> "CATTS":
    """Load CATTS from a GitHub release checkpoint.

    Parameters
    ----------
    release:
        GitHub release tag containing the checkpoint asset.
    checkpoint_name:
        Name of the checkpoint file attached to the GitHub release.
    map_location:
        Device mapping used when loading the checkpoint.
    kwargs:
        Additional keyword arguments forwarded to `CATTS.from_pretrained(...)`.

    Returns
    -------
        The loaded CATTS model.

    """
    checkpoint_url = (
        f"https://github.com/lucadellalib/catts/releases/download/"
        f"{release}/{checkpoint_name}"
    )
    checkpoint_path = os.path.join(
        torch.hub.get_dir(), "catts", release, checkpoint_name
    )
    if not os.path.exists(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.hub.download_url_to_file(
            checkpoint_url,
            checkpoint_path,
            progress=True,
        )

    model = CATTS.from_pretrained(
        checkpoint_path,
        map_location=map_location,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    model = catts()
    print(
        f"Total number of parameters/buffers: "
        f"{sum(x.numel() for x in model.state_dict().values()) / 1e6:.2f}M"
    )
