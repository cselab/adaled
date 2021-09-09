from base import TestCase

import os
import subprocess
import tempfile


def _get_num_movie_frames(path: str):
    """Read the number of video frames from a movie using ffmpeg."""
    # https://stackoverflow.com/questions/2017843/fetch-frame-count-with-ffmpeg

    cmd = 'ffprobe -v error -select_streams v:0 -count_packets ' \
          '-show_entries stream=nb_read_packets -of csv=p=0'.split() + [path]
    return int(subprocess.check_output(cmd))


def _update(k: int):
    return []


class TestPlottingAnimation(TestCase):
    def test_parallelized_animation(self):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import adaled.plotting.animation as animation

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'movie.mp4')
            fig = plt.figure()
            try:
                ani = animation.parallelized_animation(
                        path, fig, _update, range(8), workers=4, verbose=False)
            except animation._FFMPEGUnavailable:
                self.skipTest("cannot test animation, ffmpeg unavailable")
            finally:
                plt.close(fig)
            self.assertEqual(_get_num_movie_frames(path), 8)
