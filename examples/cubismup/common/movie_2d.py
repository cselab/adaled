#!/usr/bin/env python3

# Preload, because it uses a newer version of OpenMP compared to torch.
import cubismup2d

from .movie import Movie2DPlotter

if __name__ == '__main__':
    Movie2DPlotter.main()
