# Do not import cubism here.

from adaled import AdaLEDStage as Stage, TensorCollection
import adaled

import numpy as np

from typing import Optional

def extend_runtime_postprocess_data(
        post: TensorCollection,
        diagnostics: Optional[adaled.AdaLEDDiagnostics] = None,
        path: str = '') -> TensorCollection:
    post = post.map(lambda x: x)  # Shallow copy.
    meta = post['metadata']
    diff = post['micro_state_qoi'] - post['macro_qoi']
    post['__qoi_error_l1'] = adaled.cmap(np.abs, diff).sum(axis=-1)
    post['__qoi_error_l2'] = (diff ** 2).sum(axis=-1) ** 0.5
    # norm = np.mean((post['micro_state_qoi'] ** 2).sum(axis=-1) ** 0.5)
    norm = 0.079
    post['__qoi_error_norm_Fcyl'] = post['__qoi_error_l2', 'cyl_forces'] / norm
    meta['__is_macro'] = is_macro = meta['stage'] == Stage.MACRO
    meta['__is_macro_cumulative'] = \
            np.cumsum(is_macro) / np.arange(1, len(is_macro) + 1)
    post['__qoi_error_norm_Fcyl_when_macro'] = \
            np.where(is_macro, post['__qoi_error_norm_Fcyl', :, 0], np.nan)

    if len(post['transition_qoi_errors']) == 0:
        print(f"{path} removing empty transition_qoi_errors")
        del post['transition_qoi_errors']

    if diagnostics:
        per_cycle = diagnostics.per_cycle_stats.data
        per_cycle = per_cycle[:np.searchsorted(per_cycle['start_timestep'], len(diff))]
        if per_cycle['start_timestep', 1:].max() < len(diff):
            # Approximate, it is going to be smoothed out anyway.
            exectime = np.nan_to_num(meta['execution_time'])
            exectime[per_cycle['start_timestep', 1:]] -= per_cycle['stats_overhead_seconds', :-1]
            meta['__execution_time_cumulative_no_overhead'] = np.cumsum(exectime)
            if 'cmp_error' in post:
                post['cmp_error']['__v_when_macro'] = \
                        np.where(is_macro, post['cmp_error', 'v', :, 0], np.nan)
            meta['__execution_time_cumulative_no_overhead'] = exect = np.cumsum(exectime)
        else:
            print(f"{path} looks broken, probably the simulation crashed.")

    return post


def load_and_extend_runtime_postprocess_data(
        path: str,
        diagnostics: Optional[adaled.AdaLEDDiagnostics] = None) -> TensorCollection:
    post = adaled.load(path)
    post = TensorCollection(post)  # In case it is a dictionary.
    return extend_runtime_postprocess_data(post, diagnostics, path=path)
