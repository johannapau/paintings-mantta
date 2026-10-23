"""Resizing FX10 images to match true dimensions. Note: remember to flip them if mirror images!"""


import numpy as np
from scipy.ndimage import zoom

# ------------------ For typing things. ------------------
from typing import Optional, Sequence, Tuple
from numpy.typing import NDArray
from matplotlib.backend_bases import MouseEvent, Event
from matplotlib.figure import Figure





def resize_img_cube_nanfriendly(cube: NDArray, new_height: int, new_width: int, order=1):
    """
    Resize a cube spatially, keeping NaNs as NaNs. 

    - The NaN pattern is resized with nearest-neighbor, so missing regions
      keep their shape (no growing, shrinking, or filling).
    - Valid values are interpolated only from valid neighbors, so NaNs
      never contaminate them.

    Parameters
    ----------
    cube : np.ndarray
        Shape (height, width, bands). May contain NaNs.
    new_height, new_width : int
        Desired output size.
    order : int, optional
        Interpolation order for valid data (0, 1, or 3). Default 1.

    Returns
    -------
    np.ndarray
        Shape (new_height, new_width, bands), float64.
    """
    if cube.ndim != 3:
        raise ValueError("Input must be a 3D array (height, width, bands).")

    height, width, _ = cube.shape
    zf = (new_height / height, new_width / width, 1)

    valid = np.isfinite(cube)
    data = np.where(valid, cube, 0.0).astype(np.float64)

    # Where the output should be NaN: nearest-neighbor on the validity mask
    valid_out = np.asarray(zoom(valid.astype(np.uint8), zf, order=0)).astype(bool)

    # Interpolate valid data using only valid neighbors
    num: NDArray = np.array(zoom(data, zf, order=order))
    den = np.clip(np.asarray(zoom(valid.astype(np.float64), zf, order=order)), 0.0, 1.0)

    # Nearest-neighbor values as a fallback where den is ~0
    nearest = zoom(data, zf, order=0)

    out = np.full(num.shape, np.nan, dtype=np.float64)
    good = den > 1e-6

    interp: NDArray = np.where(good, num / np.where(good, den, 1.0), nearest)  # type: ignore

    out[valid_out] = interp[valid_out]
    return out





def save_painting(path, name, img):
    np.save(f'{path}/{name}.npy', img)


def main():
    paintings = ['meri_fx10.npy', 'meri_overpainted_fx10.npy', 
                 'kolea_op_fx10.npy', 'afrikka_fx10.npy', 
                 'puut_fx10.npy']

    dims = [(20,30), (20,30), (42,35), (20.5,30), (24,33) ]
    
    for idx, painting in enumerate(paintings): 
        filename = painting[:-4]
        print('Working on: ', filename)
        
        data_path = '/n/archive/labs/hsi-data/taulukuvaukset_2026/TEMP/data/FX10'
        stitched = np.load(f'{data_path}/{filename}.npy')
        w_irl, h_irl = dims[idx]

        scaler = h_irl / w_irl

        new_width  = stitched.shape[1]
        new_height = int(new_width * scaler)
        
        resized = resize_img_cube_nanfriendly(stitched, new_height, new_width)
        
        save_painting('/n/archive/labs/hsi-data/taulukuvaukset_2026/TEMP/data/resized/FX10', filename, resized)
        

if __name__ == "__main__":
    main()