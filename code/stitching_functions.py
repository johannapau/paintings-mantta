import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import patches
import matplotlib
from matplotlib.figure import Figure
from numpy.typing import NDArray
from collections.abc import Iterable
import h5py



def load_corrected_images(f: h5py.File, white_y: int, white_h: int,
                          indices: Iterable[int] | None = None,
                          flip_vertical: bool = False, vstep: int = 0) -> list[NDArray]:
    """Load and radiometrically correct images:
          refl = (raw - dark)/(white - dark)
        Returns a list [im0, im1, ...] with shape (H, W, C) each.

    Args:
        f (File): Open HDF5 file containing `img_data_*` and `black_ref_*` datasets.
        white_y (int): y coord where the white reference panel begins on the first strip.
        white_h (int): height of the white reference panel.
        indices (list[int] | None, optional): Strip indices to load. Defaults to all available.
        flip_vertical (bool, optional): True to flip each image vertically. Defaults to False.
        vstep (int, optional): Each successive strip's white reference area is lifted by this
            many pixels more than the previous one. Defaults to 0.

    Raises:
        ValueError: If a strip's white reference area falls outside the image.

    Returns:
        list[NDArray]: List of numpy arrays (reflectance data)
    """
    if indices is None:
        indices = sorted(int(k.split('_')[-1]) for k in f.keys() if k.startswith('img_data_'))

    images = []
    for pos, i in enumerate(indices):
        img: NDArray = f[f'img_data_{i}'][()]  # type: ignore
        dark_mean = f[f'black_ref_{i}'][()].mean(axis=0, keepdims=True)  # (1, W, C)  # type: ignore

        # Assuming camera was upside down --> image is upside down and white ref stripe on top
        y = white_y + pos * vstep
        if y < 0 or y + white_h > img.shape[0]: 
            raise ValueError(
                f"White reference area for strip {i} is out of bounds: "
                f"rows {y}:{y + white_h}, image height {img.shape[0]}."
            )

        white_mean = img[y:y + white_h, :, :].mean(axis=0, keepdims=True)  # (1, W, C)
        print('Current y is: ', y)

        im = (img - dark_mean) / (white_mean - dark_mean)

        if flip_vertical:
            im = np.flip(im, axis=0)

        images.append(im)

    return images


# Modified version of make_strip_canvas to work correctly with NaNs.
def make_strip_canvas_nanfriendly(images, width, offset,
                      crop_start=10, vstep=4, top_extra=16,
                      extra_last=25, reverse_for_layout=True,
                      height=None, height_align="top",
                      blend_edges=False, blend_threshold=None):
    """
    Assemble a horizontal canvas from a list of corrected images.

    New:
    - blend_edges: if True, feather overlaps horizontally with a linear ramp.
    - blend_threshold: if set (float), only blend pixels with mean abs diff <= threshold;
                       else use the newer panel's value at those pixels.

    NaN handling:
    - valid + valid -> blended normally
    - valid + NaN   -> valid
    - NaN   + valid -> valid
    - NaN   + NaN   -> NaN
    """

    assert len(images) > 0, "No images provided."

    heights = [im.shape[0] for im in images]
    widths = [im.shape[1] for im in images]
    bands = [im.shape[2] for im in images]

    assert len(set(bands)) == 1, "All images must have the same number of bands."
    C = bands[0]

    H_common = min(heights) if height is None else min(height, min(heights))

    for i, Wfull in enumerate(widths):
        if crop_start + width > Wfull:
            raise ValueError(
                f"Requested crop exceeds width for image {i}: "
                f"crop_start({crop_start}) + width({width}) > {Wfull}"
            )

    cropped = []
    for im, H in zip(images, heights):
        if H == H_common:
            imc = im
        else:
            if height_align == "top":
                imc = im[:H_common, :, :]
            elif height_align == "bottom":
                imc = im[H - H_common:, :, :]
            else:
                raise ValueError("height_align must be 'top' or 'bottom'")
        cropped.append(imc)

    ordered = cropped[::-1] if reverse_for_layout else cropped
    n = len(ordered)

    required_top = (n - 1) * vstep
    top_pad = max(top_extra, required_top)

    total_w = n * width - offset * (n - 1) + (extra_last if n >= 1 else 0)

    # CHANGED: initialize canvas with NaNs
    canvas = np.full(
        (H_common + top_pad, total_w, C),
        np.nan,
        dtype=np.float64
    )

    def nan_blend(prev_sub, curr_sub, alpha):
        """
        Blend while ignoring NaNs.

        valid + valid -> weighted blend
        valid + NaN   -> valid
        NaN   + valid -> valid
        NaN   + NaN   -> NaN
        """
        blended = prev_sub * (1.0 - alpha) + curr_sub * alpha

        prev_nan = np.isnan(prev_sub)
        curr_nan = np.isnan(curr_sub)

        blended = np.where(prev_nan & ~curr_nan, curr_sub, blended)
        blended = np.where(~prev_nan & curr_nan, prev_sub, blended)

        return blended

    def panel_origin(i):
        y0 = (n - 1 - i) * vstep
        x0 = i * (width - offset)
        if i == n - 1:
            x0 += extra_last
        return y0, x0

    y0, x0 = panel_origin(0)
    patch0 = ordered[0][:, crop_start:crop_start + width, :]
    canvas[y0:y0 + H_common, x0:x0 + width, :] = patch0

    for i in range(1, n):
        y1, x1 = panel_origin(i)
        patch = ordered[i][:, crop_start:crop_start + width, :]

        y1a, y1b = y1, y1 + H_common
        x1a, x1b = x1, x1 + width

        yp, xp = panel_origin(i - 1)
        ypa, ypb = yp, yp + H_common
        xpa, xpb = xp, xp + width

        xi1 = max(xpa, x1a)
        xi2 = min(xpb, x1b)

        yi1 = max(ypa, y1a)
        yi2 = min(ypb, y1b)

        if yi1 > y1a:
            canvas[y1a:yi1, x1a:x1b, :] = patch[:yi1 - y1a, :, :]

        if yi2 < y1b:
            canvas[yi2:y1b, x1a:x1b, :] = patch[yi2 - y1a:, :, :]

        if yi2 > yi1:

            if xi1 > x1a:
                canvas[yi1:yi2, x1a:xi1, :] = (
                    patch[yi1 - y1a:yi2 - y1a, :xi1 - x1a, :]
                )

            if xi2 < x1b:
                canvas[yi1:yi2, xi2:x1b, :] = (
                    patch[yi1 - y1a:yi2 - y1a, xi2 - x1a:, :]
                )

            if (xi2 > xi1) and (blend_edges or blend_threshold is not None):

                prev_sub = canvas[yi1:yi2, xi1:xi2, :].copy()
                curr_sub = patch[
                    yi1 - y1a:yi2 - y1a,
                    xi1 - x1a:xi2 - x1a,
                    :
                ]

                w_ov = xi2 - xi1

                alpha = np.linspace(
                    0.0, 1.0,
                    num=w_ov,
                    dtype=np.float64
                )[None, :, None]

                if blend_threshold is None:

                    blended = nan_blend(prev_sub, curr_sub, alpha)

                else:

                    diff = np.nanmean(
                        np.abs(curr_sub - prev_sub),
                        axis=2
                    )

                    mask = (
                        diff <= float(blend_threshold)
                    )[..., None]

                    blended_raw = nan_blend(
                        prev_sub,
                        curr_sub,
                        alpha
                    )

                    blended = np.where(
                        mask,
                        blended_raw,
                        curr_sub
                    )

                canvas[yi1:yi2, xi1:xi2, :] = blended

            else:

                if xi2 > xi1:
                    canvas[yi1:yi2, xi1:xi2, :] = patch[
                        yi1 - y1a:yi2 - y1a,
                        xi1 - x1a:xi2 - x1a,
                        :
                    ]

        else:

            canvas[y1a:y1b, x1a:x1b, :] = patch

    return canvas



def make_strip_canvas(images, width, offset,
                      crop_start=10, vstep=4, top_extra=16,
                      extra_last=25, reverse_for_layout=True,
                      height=None, height_align="top",
                      blend_edges=False, blend_threshold=None):
    """
    Assemble a horizontal canvas from a list of corrected images.

    New:
    - blend_edges: if True, feather overlaps horizontally with a linear ramp.
    - blend_threshold: if set (float), only blend pixels with mean abs diff <= threshold;
                       else use the newer panel's value at those pixels.

    Notes:
    - Blending is performed over the *actual 2D intersection* (x and y) between
      successive panels, which accounts for your vertical stagger.
    - Ramp increases from the left (previous panel) to the right (current panel).
    """
    assert len(images) > 0, "No images provided."

    heights = [im.shape[0] for im in images]
    widths  = [im.shape[1] for im in images]
    bands   = [im.shape[2] for im in images]
    assert len(set(bands)) == 1, "All images must have the same number of bands."
    C = bands[0]

    # Common height
    H_common = min(heights) if height is None else min(height, min(heights))

    # Width crop check
    for i, Wfull in enumerate(widths):
        if crop_start + width > Wfull:
            raise ValueError(
                f"Requested crop exceeds width for image {i}: "
                f"crop_start({crop_start}) + width({width}) > {Wfull}"
            )

    # Crop each to common height
    cropped = []
    for im, H in zip(images, heights):
        if H == H_common:
            imc = im
        else:
            if height_align == "top":
                imc = im[:H_common, :, :]
            elif height_align == "bottom":
                imc = im[H - H_common:, :, :]
            else:
                raise ValueError("height_align must be 'top' or 'bottom'")
        cropped.append(imc)

    # Layout order (to match your im3, im2, im1, im0 left→right)
    ordered = cropped[::-1] if reverse_for_layout else cropped
    n = len(ordered)

    # Ensure enough top padding for the maximum vertical stagger
    required_top = (n - 1) * vstep
    top_pad = max(top_extra, required_top)

    total_w = n * width - offset * (n - 1) + (extra_last if n >= 1 else 0)

    canvas = np.zeros((H_common + top_pad, total_w, C), dtype=np.float64)



    def panel_origin(i):
        """Return (y0, x0) for panel i in 'ordered' list."""
        y0 = (n - 1 - i) * vstep
        x0 = i * (width - offset)
        if i == n - 1:
            x0 += extra_last
        return y0, x0



    # Place first panel directly
    y0, x0 = panel_origin(0)
    patch0 = ordered[0][:, crop_start:crop_start + width, :]
    canvas[y0:y0 + H_common, x0:x0 + width, :] = patch0

    # Place the rest with optional blending
    for i in range(1, n):
        y1, x1 = panel_origin(i)
        patch = ordered[i][:, crop_start:crop_start + width, :]

        # Full current rectangle in canvas coords
        y1a, y1b = y1, y1 + H_common
        x1a, x1b = x1, x1 + width

        # Overwrite current panel completely first? NO—because we need prev data for blending.
        # We'll blend the overlap first (using a copy of the previous region), then fill non-overlap.

        # Previous panel rect (i-1)
        yp, xp = panel_origin(i - 1)
        ypa, ypb = yp, yp + H_common
        xpa, xpb = xp, xp + width

        # Intersection region (x and y)
        xi1 = max(xpa, x1a); xi2 = min(xpb, x1b)
        yi1 = max(ypa, y1a); yi2 = min(ypb, y1b)

        # Left/top non-overlap regions: write them directly
        # Top band above overlap (if any)
        if yi1 > y1a:
            canvas[y1a:yi1, x1a:x1b, :] = patch[:yi1 - y1a, :, :]
        # Bottom band below overlap (if any)
        if yi2 < y1b:
            canvas[yi2:y1b, x1a:x1b, :] = patch[yi2 - y1a:, :, :]

        # Now within the vertical intersection (if any), split horizontally
        if yi2 > yi1:
            # Left strip before horizontal overlap
            if xi1 > x1a:
                canvas[yi1:yi2, x1a:xi1, :] = patch[yi1 - y1a:yi2 - y1a, :xi1 - x1a, :]
            # Right strip after horizontal overlap
            if xi2 < x1b:
                canvas[yi1:yi2, xi2:x1b, :] = patch[yi1 - y1a:yi2 - y1a, xi2 - x1a:, :]

            # Overlap area: blend or overwrite
            if (xi2 > xi1) and (blend_edges or blend_threshold is not None):
                # pull previous subregion BEFORE modifying it
                prev_sub = canvas[yi1:yi2, xi1:xi2, :].copy()
                curr_sub = patch[yi1 - y1a:yi2 - y1a, xi1 - x1a:xi2 - x1a, :]

                w_ov = xi2 - xi1
                # horizontal ramp: 0 (prev) -> 1 (current)
                alpha = np.linspace(0.0, 1.0, num=w_ov, dtype=np.float64)[None, :, None]

                if blend_threshold is None:
                    # pure feather
                    blended = prev_sub * (1.0 - alpha) + curr_sub * alpha
                else:
                    # thresholded blend: only blend where mean abs diff <= threshold
                    diff = np.mean(np.abs(curr_sub - prev_sub), axis=2)  # (h, w)
                    mask = (diff <= float(blend_threshold))[..., None]    # (h, w, 1)
                    blended = np.where(
                        mask,
                        prev_sub * (1.0 - alpha) + curr_sub * alpha,
                        curr_sub
                    )
                canvas[yi1:yi2, xi1:xi2, :] = blended
            else:
                # No blending requested: current overwrites overlap
                if (xi2 > xi1):
                    canvas[yi1:yi2, xi1:xi2, :] = patch[yi1 - y1a:yi2 - y1a, xi1 - x1a:xi2 - x1a, :]
        else:
            # No vertical intersection at all → write the whole patch
            canvas[y1a:y1b, x1a:x1b, :] = patch

    return canvas



def convert_to_reflectance(raw: NDArray, white: NDArray, dark: NDArray) -> NDArray:
    """Converting raw data to reflectance.

    Args:
        raw (NDArray): raw data (Y, X, B)
        white (NDArray): white reference, shaped (1, X, B)
        dark (NDArray: dark reference, shaped (???)

    Returns:
        NDArray: Reflectance data, shaped (Y, X, B)
    """    
    # numpy broadcasting takes care of matching the dimensions!
    refl = (raw - dark)/(white-dark)
    
    return refl

## ------------------- IMAGE PREPROCESSING FUNCTIONS -----------------------------------



def walk_h5(name, obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, h5py.Dataset):
        print(f"{pad}📄 Dataset: {name}  shape={obj.shape}  dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{pad}📁 Group: {name}")
        for key in obj.keys():
            walk_h5(f"{name}/{key}".replace("//", "/"), obj[key], indent+1)



def plot_stripes(path: str, x_white: int, y_white: int, h_white: int,
                 alignment: str = 'N', vstep: int = 0) -> Figure:
    """Plot the image strips and visualize the white reference areas.

    Sometimes the easel/painting/camera is ever so slightly crooked,
    and one spot to clip white reference will not work with all strips.

    Args:
        path (str): Path to the HDF5 file containing the `img_data_*` strips.
        x_white (int): Left edge (x) of the white reference rectangle.
        y_white (int): Top edge (y) of the white reference rectangle on the first strip.
        h_white (int): Height of the white reference rectangle.
        alignment (str, optional): 'N' (top) or 'S' (bottom) anchor. Defaults to 'N'.
        vstep (int, optional): Each successive strip's rectangle is lifted by this
            many pixels more than the previous one. Defaults to 0.

    Raises:
        ValueError: If alignment is not 'N' or 'S'.

    Returns:
        Figure: The matplotlib figure with all strips.
    """
    if alignment not in ('N', 'S'):
        raise ValueError("alignment must be either 'N' (top/north) or 'S' (bottom/south).")

    with h5py.File(path, "r") as f:
        keys = sorted(
            (k for k in f.keys() if k.startswith("img_data_")),
            key=lambda k: int(k.split('_')[-1]),
        )
        cubes: list[NDArray] = [f[k][:] for k in keys]  # read everything in one open  # type: ignore

    cols = len(keys)
    fig, axes = plt.subplots(1, cols, dpi=100, squeeze=False)
    axes = axes[0]  # squeeze=False keeps this indexable even for a single strip

    for idx, cube in enumerate(cubes):
        # Camera mounted upside down --> image upside down and white strip on top if the strips
        y = y_white + idx * vstep

        rect = patches.Rectangle(
            (x_white, y),
            width=cube.shape[1] - x_white,
            height=h_white,
            linewidth=1,
            edgecolor='red',
            facecolor='none',
        )

        axes[idx].imshow(cube[:, :, 100])
        axes[idx].set_anchor(alignment)
        axes[idx].axis('off')
        axes[idx].add_patch(rect)

    fig.tight_layout()
    return fig