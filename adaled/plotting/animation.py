from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import tqdm

from typing import Optional, Sequence
import multiprocessing
import os
import subprocess
import tempfile
import time

# For tests only.
class _FFMPEGUnavailable(NotImplementedError):
    pass

def ffmpeg_concatenate(
        input_paths: Sequence[str],
        output_path: str,
        tmppath: str,
        verbose: bool = True):
    """Concatenate movies using ffmpeg."""
    # https://stackoverflow.com/questions/7333232/how-to-concatenate-two-mp4-files-using-ffmpeg
    with open(tmppath, 'w') as f:
        for path in input_paths:
            path = os.path.abspath(path)
            f.write(f'file \'{path}\'\n')

    try:
        cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', tmppath,
               '-loglevel', 'error', '-c', 'copy', '-y',
               os.path.abspath(output_path)]
        if verbose:
            cmd.append('-stats')
            print("Concatenating...")
        subprocess.check_output(cmd)
    finally:
        os.remove(tmppath)


def _run(processes: Sequence[multiprocessing.Process]):
    try:
        for p in processes:
            p.start()
        while True:
            alive_cnt = 0
            for p in processes:
                alive_cnt += int(p.is_alive())
                if p.exitcode is not None:
                    if p.exitcode > 0:
                        raise RuntimeError("subprocess non-zero exit code: "
                                           + str(p.exitcode))
            if alive_cnt == 0:
                break
            time.sleep(0.1)
    except:
        for p in processes:
            if p.is_alive():
                p.kill()
        raise

    for p in processes:
        p.join()
        p.close()


def parallelized_animation(path: str, fig, func, frames, *args,
                           workers: Optional[int] = None,
                           verbose: bool = True,
                           bitrate: int = -1, **kwargs):
    """Split and render an animation with `workers` processes.

    For convenience, by specifying `png` extension, this function can also be
    used to plot individual frames. In that case, the `path` may be a format
    string with a `frame` argument, e.g. `movie-{frame:07d}.png`.
    """
    frames = list(frames)
    if path.endswith('.png') or path.endswith('pdf'):
        for frame in frames:
            frame_path = path.format(frame=frame)
            print(f"Saving movie frame #{frame} to {frame_path}.")
            try:
                func(frame)
            except IndexError as e:
                print(f"WARNING: IndexError while plotting frame #{frame}: {e}")
            else:
                fig.savefig(frame_path, bbox_inches='tight')
        return

    if path.endswith('.mp4') and not mpl.animation.writers.is_available('ffmpeg'):
        raise _FFMPEGUnavailable("ffmpeg unavailable, cannot render a movie")

    if not workers:
        workers = (os.cpu_count() + 3) // 4

    workers = min(workers, len(frames) // 2)

    if verbose:
        print(f"Animating {len(frames)} frames with {workers} workers.")

    counter = multiprocessing.Value('i', 0)
    finished = multiprocessing.Value('i', 0)

    _, ext = os.path.splitext(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = os.path.abspath(tmpdir)
        paths = [os.path.join(tmpdir, f'part-{i:03d}{ext}')
                 for i in range(workers)]

        def render(i: int):
            seen_frames = set()
            def wrapped_frame_func(idx: int):
                if idx not in seen_frames:
                    counter.value += 1
                    seen_frames.add(idx)
                return func(idx)

            n = len(frames)
            part = frames[n * i // workers : n * (i + 1) // workers]
            ani = FuncAnimation(fig, wrapped_frame_func, *args,
                                frames=part, **kwargs)
            ani.save(paths[i], writer='ffmpeg', extra_args=['-threads', '1'],
                     bitrate=bitrate)
            finished.value += 1

        def progress():
            with tqdm.tqdm(total=len(frames)) as pbar:
                while finished.value < workers:
                    pbar.update(counter.value - pbar.n)
                    time.sleep(0.3)

        # We do not use multiprocessing.Pool because it tries to pickle the
        # update function `func`.
        mp = multiprocessing.get_context('fork') 
        ps = [mp.Process(target=render, args=(i,))
              for i in range(workers)]
        if verbose:
            ps.append(mp.Process(target=progress))
        _run(ps)

        listpath = os.path.join(tmpdir, 'partlist.txt')
        ffmpeg_concatenate(paths, path, listpath, verbose=verbose)
