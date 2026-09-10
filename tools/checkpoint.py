"""Reading a training checkpoint, in one place.

`tools/train.py` writes them, `tools/quantise.py` converts them and `tools/evalnet.py`
scores them. All three have to agree about how the file is opened, and one of them getting
it wrong shows up as a crash in the middle of a long run.
"""

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path) -> dict[str, Any]:
    """A training checkpoint, read without turning off `weights_only`.

    Runs before this one recorded `torch.__version__` as the `TorchVersion` object it
    really is rather than as a string, and that type is not on torch.load's allowlist. It
    is allowlisted here rather than reading those files with `weights_only=False`, which
    would execute whatever a checkpoint happened to contain.
    """
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    blob: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return blob
