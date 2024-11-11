#!/usr/bin/env python

# =====================
# Define Olympus colors
# =====================

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Use numpy array for better performance
_olympus_reference_colors = np.array([
    "#08294C",
    "#75BBE1",
    "#D4E9F4",
    "#F2F2F2",
    "#F7A4D4",
    "#F75BB6",
    "#EB0789",
])

# Add N parameter for better color interpolation
_olympus_cmap = LinearSegmentedColormap.from_list(
    "olympus", _olympus_reference_colors, N=256
)
_olympus_cmap_r = LinearSegmentedColormap.from_list(
    "olympus_r", _olympus_reference_colors[::-1], N=256
)

# Register the colormaps using the new API
colormaps.register(_olympus_cmap)
colormaps.register(_olympus_cmap_r)


def get_olympus_colors(n):
    """Get n evenly spaced colors from the Olympus colormap.
    
    Args:
        n (int): Number of colors to return
        
    Returns:
        list: RGBA colors from the Olympus colormap
    """
    olympus_cmap = colormaps["olympus"]
    return olympus_cmap(np.linspace(0, 1, n))
