#!/usr/bin/env python3

from adaled import TensorCollection
from adaled.nn.loss import weighted_mse_losses
from adaled.postprocessing.record import LazyRecordLoader
from adaled.utils.buffer import DynamicArray
import adaled

import numpy as np
import torch

import glob

def compute_postprocess_data() -> TensorCollection:
    paths = sorted(glob.glob('record-0*-0*.h5'))

    metadata = adaled.DynamicArray()
    mse_abs = adaled.DynamicArray()
    mse_rel = adaled.DynamicArray()

    for path in paths:
        print(f"Processing {path}...")
        record = adaled.load(path)
        x = record['fields', 'simulations', 'x']
        micro = torch.from_numpy(x['micro'])
        macro = torch.from_numpy(x['macro'])
        partial_abs = weighted_mse_losses(
                computed=macro, expected=micro, weight=None, relative=False)
        partial_rel = weighted_mse_losses(
                computed=macro, expected=micro, weight=None, relative=True)

        meta = record['fields', 'metadata']
        metadata.extend(TensorCollection(timestep=meta['timestep'], stage=meta['stage']))
        mse_abs.extend(partial_abs.numpy())
        mse_rel.extend(partial_rel.numpy())

    return TensorCollection({
        'metadata': metadata.data,
        'errors': {
            'mse_abs': mse_abs.data,
            'mse_rel': mse_rel.data,
        },
    })


def main():
    data = compute_postprocess_data()
    adaled.save(data, 'postprocessed.pt')


if __name__ == '__main__':
    main()
