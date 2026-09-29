"""Flatten weights/net.npz to a little-endian blob, which the build embeds in the binary so
it carries its network with it.

The npz is deflate-compressed, and reading zip in the engine means either zlib at runtime or a zip
parser in the engine. A flat blob written here needs neither.

Blob layout, every field little-endian:

    b"NNUE", uint32 version, uint32 L1, uint32 buckets, int32 qa, int32 qb, int32 scale
    int16 ft_weight[768][L1], int16 ft_bias[L1], int16 out_weight[buckets][2 * L1],
    int32 out_bias[buckets]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NET_MAGIC = b"NNUE"
NET_VERSION = 1


def export_net(source: Path, destination: Path) -> None:
    with np.load(source) as data:
        ft_weight = data["ft_weight"].astype("<i2")
        ft_bias = data["ft_bias"].astype("<i2")
        out_weight = data["out_weight"].astype("<i2")
        out_bias = data["out_bias"].astype("<i4")
        qa, qb, scale = int(data["qa"]), int(data["qb"]), int(data["scale"])
    hidden = ft_bias.shape[0]
    buckets = out_bias.shape[0]
    if ft_weight.shape != (768, hidden) or out_weight.shape != (buckets, 2 * hidden):
        raise ValueError(f"{source} has an unexpected shape: {ft_weight.shape}, {out_weight.shape}")
    header = NET_MAGIC + struct.pack("<IIIiii", NET_VERSION, hidden, buckets, qa, qb, scale)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out:
        out.write(header)
        for array in (ft_weight, ft_bias, out_weight, out_bias):
            out.write(np.ascontiguousarray(array).tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "weights" / "net.npz")
    arguments = parser.parse_args()

    export_net(arguments.source, arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
