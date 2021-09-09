#!/usr/bin/env python3

# Preload, because it uses a newer version of OpenMP compared to torch.
import libcubismup3d

from .movie import Movie3DPlotter

if __name__ == '__main__':
    Movie3DPlotter.main()
