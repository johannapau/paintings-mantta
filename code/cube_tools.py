import os
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector, TextBox, RadioButtons, LassoSelector, Slider
from matplotlib.path import Path
import itertools
from matplotlib.backend_bases import MouseButton


# ------------------ For typing things. ------------------
from typing import Optional, Sequence, Tuple
from numpy.typing import NDArray
from matplotlib.backend_bases import MouseEvent, Event
from matplotlib.figure import Figure


class CubeClipper:
    """
    Simple clipping tool for a hyperspectral cube.

    - Shows one band of the cube.
    - Drag a rectangle on the image
    - Type a name in the box, press "Save": the region is cut out of the
      FULL cube (all bands) and saved as <name>.npy with shape (Y, X, B).

    Parameters
    ----------
    cube : np.ndarray
        An array (rows, cols, bands). 
    band : int
        Band index to display.
    out_dir : str
        Folder where clips are saved (created if missing).
    """

    def __init__(self, cube: NDArray, band: int = 0, out_dir: str = "."):
        self.cube = cube
        self.rows, self.cols, self.bands = cube.shape
        self.band = int(np.clip(band, 0, self.bands - 1))
        self.out_dir = out_dir
        self.sel = None  # (r0, r1, c0, c1), end exclusive

        # ---- band image, robust to NaN/inf and odd value ranges
        img = np.asarray(self.cube[..., self.band], dtype=float)
        if np.all(np.isnan(img)):
            raise ValueError(f"Band {self.band} contains no finite values.")

        # ---- figure ------------
        self.fig: Figure = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_axes((0.05, 0.16, 0.9, 0.78))
        self.im = self.ax.imshow(img, vmin=0, vmax=1,  ## Reflectance values (should be) between 0 and 1!
                                 origin="upper")
        self.ax.set_title(f"Band {self.band}")

        ax_name = self.fig.add_axes((0.12, 0.07, 0.40, 0.05))
        ax_btn = self.fig.add_axes((0.55, 0.07, 0.15, 0.05))
        self.name_box = TextBox(ax_name, "Name: ", initial="clip")
        self.btn = Button(ax_btn, "Save")
        self.btn.on_clicked(self._on_save)
        self.status = self.fig.text(0.05, 0.02, "No selection yet.", fontsize=10)

        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=False, button=[MouseButton.LEFT],
            minspanx=2, minspany=2, spancoords="pixels", interactive=True,
            props=dict(facecolor="none", edgecolor="red", linewidth=1),
        )
        plt.show()

    # ------------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        self.status.set_text(text)
        self.fig.canvas.draw_idle()

    def _on_select(self, eclick: MouseEvent, erelease: MouseEvent) -> None:
        x0, x1, y0, y1 = self.selector.extents
        c0 = int(np.floor(min(x0, x1) + 0.5)); c1 = int(np.ceil(max(x0, x1) + 0.5))
        r0 = int(np.floor(min(y0, y1) + 0.5)); r1 = int(np.ceil(max(y0, y1) + 0.5))
        r0, r1 = (int(v) for v in np.clip([r0, r1], 0, self.rows))
        c0, c1 = (int(v) for v in np.clip([c0, c1], 0, self.cols))
        if r1 <= r0 or c1 <= c0:
            self.sel = None
            self._set_status("Selection is empty.")
            return
        self.sel = (r0, r1, c0, c1)
        self._set_status(f"Selected rows {r0}:{r1}, cols {c0}:{c1}  ->  "
                         f"{r1 - r0} x {c1 - c0} x {self.bands}")

    def _on_save(self, event: Event | None = None) -> None:
        if self.sel is None:
            self._set_status("Nothing selected. Draw a rectangle first.")
            return
        name = self.name_box.text.strip() or "clip"
        if not name.lower().endswith(".npy"):
            name += ".npy"
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(self.out_dir, name))
        r0, r1, c0, c1 = self.sel
        sub = np.array(self.cube[r0:r1, c0:c1, :])  # copy out of the (memmapped) cube
        np.save(path, sub)
        self._set_status(f"Saved {sub.shape} -> {path}")

        print('Clipping saved!')

    def get_clip(self) -> NDArray | None:
        """Current selection as an array (all bands), or None."""
        if self.sel is None:
            return None
        r0, r1, c0, c1 = self.sel

        return np.array(self.cube[r0:r1, c0:c1, :])


# --- Usage ---
# %matplotlib widget
# clipper = CubeClipper(cube, band=30, out_dir="clips")


def _to_rgb(cube, rgb_bands=(60, 30, 10), eps=1e-6):
    """Make a simple RGB from three band indices (R,G,B)."""
    r, g, b = [np.clip(cube[..., i], 0, None) for i in rgb_bands]
    stack = np.stack([r, g, b], axis=-1).astype(float)
    # per-channel min-max normalize for display
    vmin = stack.reshape(-1, 3).min(axis=0)
    vmax = stack.reshape(-1, 3).max(axis=0)
    rgb = (stack - vmin) / (vmax - vmin + eps)
    rgb = rgb**0.5  # gentle gamma for nicer view

    return np.clip(rgb, 0, 1)



class SpectralExplorer:
    """
    Interactive spectrum explorer (multi-select):
      - Point mode: each click adds a pixel spectrum with its own color.
      - Lasso mode: each lasso adds mean±std band with its own color; polygon outlined on image.
      - Clear button removes all selections.
    """
    def __init__(self, cube, wavelengths=None, rgb_bands=(60, 30, 10), title=None):
        assert cube.ndim == 3, "cube must be (rows, cols, bands)"
        self.cube = cube
        self.wl = wavelengths if wavelengths is not None else np.arange(cube.shape[2])
        self.rows, self.cols, self.bands = cube.shape
        self.rgb = _to_rgb(cube, rgb_bands=rgb_bands)
        self.mode = "Point"

        # Storage for multiple selections
        self.point_art = []    # list of dicts: {'r':..,'c':..,'line':..,'scatter':..,'color':..}
        self.lasso_art = []    # list of dicts: {'idx':np.array,'line':..,'fill':..,'polyline':..,'color':..,'n':..}

        # Color cycle
        base_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
        self.color_cycle = itertools.cycle(base_colors)

        # --- layout
        self.fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        gs = self.fig.add_gridspec(2, 5, height_ratios=[20, 1], width_ratios=[1.2, 1.2, 0.25, 1.7, 1.7])
        self.ax_img   = self.fig.add_subplot(gs[0, 0:2])
        self.ax_spec  = self.fig.add_subplot(gs[0, 3:5])
        self.ax_mode  = self.fig.add_subplot(gs[0, 2])
        self.ax_clear = self.fig.add_subplot(gs[1, 0])
        self.ax_info  = self.fig.add_subplot(gs[1, 3:5]); self.ax_info.axis("off")

        # --- image
        self.im = self.ax_img.imshow(self.rgb, origin='upper')
        self.ax_img.set_title(title or "Click (Point) or Lasso (drag) on the image")
        self.ax_img.set_xticks([]); self.ax_img.set_yticks([])

        # --- spectrum axes
        self.ax_spec.set_title("Spectrum")
        self.ax_spec.set_xlabel("Wavelength / band")
        self.ax_spec.set_ylabel("Reflectance / DN")

        # --- widgets
        self.radio = RadioButtons(self.ax_mode, ('Point', 'Lasso'))
        self.radio.on_clicked(self._on_mode_change)
        self.btn_clear = Button(self.ax_clear, "Clear")
        self.btn_clear.on_clicked(self._on_clear)

        # --- event connections
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.lasso = LassoSelector(self.ax_img, onselect=self._on_lasso, useblit=True)

        # Legend placeholder
        self._refresh_legend()
        plt.show()

    # ---------- callbacks ----------
    def _on_mode_change(self, label):
        self.mode = label
    
    def _on_clear(self, event=None):
        # remove point selections
        for p in self.point_art:
            self._safe_remove(p.get('line'))
            self._safe_remove(p.get('scatter'))
        # remove lasso selections
        for L in self.lasso_art:
            self._safe_remove(L.get('line'))
            self._safe_remove(L.get('fill'))      # fill_between -> PolyCollection
            self._safe_remove(L.get('polyline'))  # outline on image
        self.point_art.clear()
        self.lasso_art.clear()

        # update UI
        self._set_info("Cleared selections.")
        # nuke legend if present
        leg = self.ax_spec.get_legend()
        if leg is not None:
            leg.remove()

        self.ax_spec.relim(); self.ax_spec.autoscale_view()
        self.fig.canvas.draw_idle()



    def _on_click(self, event):
        if event.inaxes != self.ax_img or self.mode != "Point":
            return
        if event.xdata is None or event.ydata is None:
            return
        c = int(round(event.xdata))
        r = int(round(event.ydata))
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return

        color = next(self.color_cycle)
        spec = self.cube[r, c, :].astype(float)
        # spectrum line
        (line,) = self.ax_spec.plot(self.wl, spec, lw=1.8, color=color, label=f"Pixel (r={r}, c={c})")
        # matching marker on image
        scat = self.ax_img.scatter([c], [r], s=45, marker='x', c=[color])

        self.point_art.append({'r': r, 'c': c, 'line': line, 'scatter': scat, 'color': color})
        self._set_info(f"Added pixel (r={r}, c={c}). Total points: {len(self.point_art)}")
        self._refresh_legend()
        self.ax_spec.relim(); self.ax_spec.autoscale_view()
        self.fig.canvas.draw_idle()

    def _on_lasso(self, verts):
        if self.mode != "Lasso":
            return
        path = Path(verts)
        # grid of pixel centers
        yy, xx = np.mgrid[0:self.rows, 0:self.cols]
        coords = np.column_stack((xx.ravel(), yy.ravel()))
        mask = path.contains_points(coords).reshape(self.rows, self.cols)
        idx = np.where(mask.ravel())[0]
        n = idx.size
        if n == 0:
            self._set_info("Lasso contained no pixels.")
            return

        color = next(self.color_cycle)
        X = self.cube.reshape(-1, self.bands).astype(float)
        S = X[idx, :]
        mean, std = S.mean(axis=0), S.std(axis=0)

        (line,) = self.ax_spec.plot(self.wl, mean, lw=2.0, color=color, label=f"Lasso #{len(self.lasso_art)+1} (n={n})")
        fill = self.ax_spec.fill_between(self.wl, mean-std, mean+std, alpha=0.22, color=color)

        # Draw polygon outline on the image
        px = np.array(verts)[:, 0]
        py = np.array(verts)[:, 1]
        polyline, = self.ax_img.plot(px, py, color=color, lw=1.0)

        self.lasso_art.append({'idx': idx, 'line': line, 'fill': fill, 'polyline': polyline, 'color': color, 'n': n})
        self._set_info(f"Lasso selected {n} pixels. Total lassos: {len(self.lasso_art)}")
        self._refresh_legend()
        self.ax_spec.relim(); self.ax_spec.autoscale_view()
        self.fig.canvas.draw_idle()

    # ---------- helpers ----------
    def _refresh_legend(self):
        # Build legend from current artists
        handles = []
        labels = []
        for p in self.point_art:
            handles.append(p['line']); labels.append(p['line'].get_label())
        for L in self.lasso_art:
            handles.append(L['line']); labels.append(L['line'].get_label())
        if handles:
            lgd = self.ax_spec.legend(handles, labels, fontsize=8, frameon=True)
        else:
            leg = self.ax_spec.get_legend()
            if leg is not None:
                leg.remove()
        self.fig.canvas.draw_idle()

    def _set_info(self, text):
        self.ax_info.clear(); self.ax_info.axis("off")
        self.ax_info.text(0, 0.7, text, fontsize=10)
    
    def _safe_remove(self, artist):
        """Remove a Matplotlib artist or a list/tuple of artists without raising."""
        if artist is None:
            return
        try:
            # single artist (line, scatter, PolyCollection, etc.)
            artist.remove()
        except Exception:
            # maybe it's an iterable of artists
            try:
                for a in artist:
                    try:
                        a.remove()
                    except Exception:
                        pass
            except Exception:
                pass

# --- Usage ---
# %matplotlib widget
# explorer = SpectralExplorer(cube_ref, wavelengths=wavelengths, rgb_bands=(60, 30, 10), title="RGB preview")



class BandSurfer:
    """
    Interactive band browser for hyperspectral cubes.

    Features
    --------
    - Slider to move between bands (or use ←/→ keys).
    - Prev / Next buttons.
    - Play/Pause animation through bands.
    - Stretch options: Min-Max, P2–P98, P1–P99, Z-score.
    - Gamma correction slider.
    - Histogram of the current band with stretch markers.
    - Optional wavelength labels in titles.

    Notes
    -----
    - In a Jupyter notebook, enable an interactive backend first:
        %matplotlib notebook
      or
        %matplotlib widget   # requires ipympl

    - For large cubes, histogram/percentiles are computed on a random sample.
    """

    def __init__(
        self,
        cube: np.ndarray,                      # (rows, cols, bands)
        wavelengths: Optional[Sequence[float]] = None,
        cmap: str = "gray",
        init_band: int = 0,
        sample_for_hist: int = 80_000,         # pixels to sample for hist/percentiles
        play_interval_ms: int = 200,
    ):
        assert cube.ndim == 3, "cube must be (rows, cols, bands)"
        self.cube = cube
        self.rows, self.cols, self.bands = cube.shape
        self.wl = np.asarray(wavelengths) if wavelengths is not None else None
        self.cmap = cmap
        self.sample_for_hist = int(sample_for_hist)
        self.play_interval_ms = int(play_interval_ms)

        # cache for stretch stats per band per method
        self._cache = {}  # key: (band, method) -> (vmin, vmax) or (mean,std) for zscore

        # state
        self.band = int(np.clip(init_band, 0, self.bands-1))
        self.stretch_method = "P2–P98"  # default
        self.gamma = 1.0
        self.playing = False

        # pre-sample indices for speed (consistent sampling)
        rng = np.random.default_rng(0)
        n_pix = self.rows * self.cols
        if n_pix > self.sample_for_hist:
            self.sample_idx = rng.choice(n_pix, size=self.sample_for_hist, replace=False)
        else:
            self.sample_idx = np.arange(n_pix)

        # figure layout
        self.fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        gs = self.fig.add_gridspec(3, 6, height_ratios=[16, 1, 1], width_ratios=[1.5, 1.5, 0.1, 1.5, 1.5, 1.4])

        self.ax_img  = self.fig.add_subplot(gs[0, 0:3])   # big image
        self.ax_hist = self.fig.add_subplot(gs[0, 3:6])   # histogram
        self.ax_slider = self.fig.add_subplot(gs[1, 0:6]) # band slider

        self.ax_prev = self.fig.add_subplot(gs[2, 0])
        self.ax_next = self.fig.add_subplot(gs[2, 1])
        self.ax_play = self.fig.add_subplot(gs[2, 2])
        self.ax_gamma = self.fig.add_subplot(gs[2, 3])
        self.ax_stretch = self.fig.add_subplot(gs[2, 4])
        self.ax_cmap = self.fig.add_subplot(gs[2, 5])

        # image artist
        img0 = self._get_band_image(self.band)
        self.im = self.ax_img.imshow(img0, cmap=self.cmap, origin="upper")
        self.ax_img.set_xticks([]); self.ax_img.set_yticks([])
        self._set_img_title()

        # histogram artist
        (self.hist_line,) = self.ax_hist.plot([], [], lw=1.5)
        (self.vmin_line,) = self.ax_hist.plot([], [], "r--", lw=1.0)
        (self.vmax_line,) = self.ax_hist.plot([], [], "r--", lw=1.0)
        self.ax_hist.set_title("Band Histogram")
        self.ax_hist.set_xlabel("Intensity")
        self.ax_hist.set_ylabel("Count")

        # band slider
        self.slider = Slider(self.ax_slider, "Band", 0, self.bands-1, valinit=self.band, valstep=1)
        self.slider.on_changed(self._on_slider)

        # buttons
        self.btn_prev = Button(self.ax_prev, "◀ Prev")
        self.btn_prev.on_clicked(lambda evt: self._step(-1))
        self.btn_next = Button(self.ax_next, "Next ▶")
        self.btn_next.on_clicked(lambda evt: self._step(+1))
        self.btn_play = Button(self.ax_play, "▶ Play")
        self.btn_play.on_clicked(self._toggle_play)

        # gamma slider (log-ish range)
        self.gamma_slider = Slider(self.ax_gamma, "Gamma", 0.2, 3.0, valinit=self.gamma)
        self.gamma_slider.on_changed(self._on_gamma)

        # stretch radio
        self.radio_stretch = RadioButtons(self.ax_stretch, ("Min–Max", "P2–P98", "P1–P99", "Z-score"))
        self.radio_stretch.on_clicked(self._on_stretch)

        # cmap radio (a few useful ones)
        self.radio_cmap = RadioButtons(self.ax_cmap, ("gray", "viridis", "magma", "inferno"))
        self.radio_cmap.on_clicked(self._on_cmap)

        # timer for play
        self._timer = self.fig.canvas.new_timer(interval=self.play_interval_ms)
        self._timer.add_callback(lambda: self._step(+1, from_timer=True))

        # keyboard for ← / →
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # initial histogram
        self._update_histogram()
        plt.show()

    # --------- UI callbacks ----------
    def _on_slider(self, val):
        self.band = int(val)
        self._refresh_band()

    def _on_gamma(self, val):
        self.gamma = float(val)
        self._refresh_image_only()

    def _on_stretch(self, label):
        self.stretch_method = label
        self._refresh_image_only()
        self._update_histogram()

    def _on_cmap(self, label):
        self.cmap = label
        self.im.set_cmap(self.cmap)
        self.fig.canvas.draw_idle()

    def _on_key(self, evt):
        if evt.key in ("left", "a"):
            self._step(-1)
        elif evt.key in ("right", "d"):
            self._step(+1)
        elif evt.key in (" ", "p"):
            self._toggle_play()

    def _toggle_play(self, evt=None):
        self.playing = not self.playing
        self.btn_play.label.set_text("❚❚ Pause" if self.playing else "▶ Play")
        if self.playing:
            self._timer.start()
        else:
            self._timer.stop()
        self.fig.canvas.draw_idle()

    # --------- helpers ----------
    def _step(self, delta, from_timer=False):
        new_band = (self.band + int(delta)) % self.bands
        self.band = new_band
        # update slider without triggering recursion
        self.slider.set_val(self.band)
        # slider callback will refresh; when called from timer, the slider event already fired
        if from_timer:
            # ensure visuals are up to date even if slider backend throttled
            self._refresh_band()

    def _refresh_band(self):
        self._set_img_title()
        self._refresh_image_only()
        self._update_histogram()

    def _set_img_title(self):
        if self.wl is not None and 0 <= self.band < len(self.wl):
            self.ax_img.set_title(f"Band {self.band}  (λ ≈ {self.wl[self.band]:.2f})")
        else:
            self.ax_img.set_title(f"Band {self.band}")

    def _get_band_image(self, band: int):
        """Return display-ready 2D image for the given band, applying stretch+gamma."""
        img = self.cube[..., band].astype(float)
        vmin, vmax, mode = self._get_stretch_params(band, self.stretch_method, img)
        if mode == "z":  # z-score to 0..1 via mean±3σ window
            mean, std = vmin, vmax
            if std <= 0:
                disp = np.zeros_like(img)
            else:
                disp = (img - (mean - 3*std)) / (6*std)
        else:
            if vmax <= vmin:
                disp = np.zeros_like(img)
            else:
                disp = (img - vmin) / (vmax - vmin)
        # gamma
        disp = np.clip(disp, 0, 1) ** self.gamma
        return disp

    def _get_stretch_params(self, band, method, img2d=None):
        """
        Returns (vmin, vmax, mode) where mode is 'range' or 'z'.
        For 'z', vmin=mean and vmax=std (used as parameters).
        """
        key = (band, method)
        if key in self._cache:
            return self._cache[key] + ("z" if method == "Z-score" else "range",)

        if img2d is None:
            img2d = self.cube[..., band].astype(float)
        flat = img2d.ravel()
        flat = flat[self.sample_idx]  # sample for speed

        if method == "Min–Max":
            vmin, vmax = np.nanmin(flat), np.nanmax(flat)
            self._cache[key] = (vmin, vmax)
            return vmin, vmax, "range"
        elif method == "P2–P98":
            vmin, vmax = np.nanpercentile(flat, [2, 98])
            self._cache[key] = (vmin, vmax)
            return vmin, vmax, "range"
        elif method == "P1–P99":
            vmin, vmax = np.nanpercentile(flat, [1, 99])
            self._cache[key] = (vmin, vmax)
            return vmin, vmax, "range"
        elif method == "Z-score":
            mean = np.nanmean(flat)
            std  = float(np.nanstd(flat))
            if not np.isfinite(std) or std == 0:
                std = 1.0
            self._cache[key] = (mean, std)
            return mean, std, "z"
        else:
            # fallback
            vmin, vmax = np.nanpercentile(flat, [2, 98])
            self._cache[key] = (vmin, vmax)
            return vmin, vmax, "range"

    def _refresh_image_only(self):
        disp = self._get_band_image(self.band)
        self.im.set_data(disp)
        self.fig.canvas.draw_idle()

    def _update_histogram(self, bins: int = 128):
        img = self.cube[..., self.band].astype(float)
        data = img.ravel()[self.sample_idx]
        counts, edges = np.histogram(data[~np.isnan(data)], bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.hist_line.set_data(centers, counts)

        # stretch markers
        vmin, vmax, mode = self._get_stretch_params(self.band, self.stretch_method, img)
        if mode == "z":
            # draw mean±3σ lines
            mean, std = vmin, vmax
            a, b = mean - 3*std, mean + 3*std
            self.vmin_line.set_data([a, a], np.array([0, counts.max() if counts.size else 1]))
            self.vmax_line.set_data([b, b], np.array([0, counts.max() if counts.size else 1]))
        else:
            self.vmin_line.set_data([vmin, vmin], np.array([0, counts.max() if counts.size else 1]))
            self.vmax_line.set_data([vmax, vmax], np.array([0, counts.max() if counts.size else 1]))

        self.ax_hist.relim(); self.ax_hist.autoscale_view()
        self.fig.canvas.draw_idle()



# --- Usage --- 
# %matplotlib widget
# surfer = BandSurfer(cube, wavelengths=wls)