"""Short local training throughput probe. Saves timings only, never model weights."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import dataset  # noqa: E402
from tools.train import Network, batches  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(20260909)
    device = torch.device('cuda')
    source = dataset.load(ROOT / 'data' / 'train.bin')
    records = np.array(source[:262144], copy=True)
    results = []
    for batch_size in (4096, 16384):
        model = Network(512, 8).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        times = []
        for repeat in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for white, black, offsets, stm, bucket, score in batches(
                records, batch_size, device, 8, np.random.default_rng(20260909 + repeat)
            ):
                predicted = torch.sigmoid(model(white, black, offsets, stm, bucket))
                loss = ((predicted - torch.sigmoid(score / 400))**2).mean()
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                model.clamp_()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        results.append({'batch': batch_size, 'positions': len(records), 'seconds': times,
                        'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
    (ROOT / 'audit' / 'train-probe.json').write_text(json.dumps({
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'results': results
    }, indent=2) + '\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
