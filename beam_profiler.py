import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from mpl_toolkits.mplot3d import Axes3D
import imageio.v3 as iio
from scipy.ndimage import median_filter, uniform_filter
from scipy.optimize import curve_fit


class Image:

    def __init__(self, path_file: str):
        "Initialize the image by loading it from the provided path"
        self.path = path_file
        try:
            self.image = iio.imread(path_file)
            self.shape = self.image.shape
            self.ny, self.nx = self.shape[:2]
            print(f"The image has N_x = {self.nx} pixels along x and N_y = {self.ny} pixels along y")
        except Exception as e:
            print(f"Error while loading with imageio: {e}")
            self.image = None

    @classmethod
    def from_array(cls, array, path=None):
        obj = cls.__new__(cls)
        obj.path = path
        obj.image = array
        obj.shape = array.shape
        obj.ny, obj.nx = array.shape[:2]
        return obj

    def plot(self, plot_3D: bool = True):
        """
        Plot image:
        - 2D greyscale
        - 3D surface intensity
        """
        if self.image is None:
            raise RuntimeError("No image loaded")

        if plot_3D:
            fig = plt.figure(figsize=(14, 6), dpi=300)

            # 2D Image
            ax1 = fig.add_subplot(1, 2, 1)
            im = ax1.imshow(self.image, cmap="gray")
            ax1.set_title("Intensity map (2D)")
            ax1.axis("off")
            plt.colorbar(im, ax=ax1, fraction=0.046)

            # 3D Surface
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")

            x = np.arange(self.nx)
            y = np.arange(self.ny)

            X, Y = np.meshgrid(x, y)

            surf = ax2.plot_surface(
                X, Y, self.image,
                cmap="viridis",
                linewidth=0,
                antialiased=True
            )

            ax2.set_title("Intensity profile (3D)")
            ax2.set_xlabel("x [px]")
            ax2.set_ylabel("y [px]")
            ax2.set_zlabel("Intensity")
            fig.colorbar(surf, ax=ax2, fraction=0.046)

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            plt.close()

        else:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            ax.imshow(self.image, cmap="gray")
            ax.axis("off")

            plt.show(block=False)
            plt.pause(0.1)
            plt.close()

    def subtract_dark(
        self,
        dark: "Image",
        *,
        apply_dark: bool = True,
        low_clip: float | None = None,
        low_clip_percentile: float | None = None,
        clip_negative: bool = True,
        debug_plot: bool = True
    ) -> "Image":

        if not apply_dark:
            return self

        if self.image is None or dark.image is None:
            raise RuntimeError("Image or dark frame not loaded")

        img = self.image
        dimg = dark.image

        img_f = img.astype(np.float64)
        dimg_f = dimg.astype(np.float64)

        if img_f.shape == dimg_f.shape:
            dark_used = dimg_f
        else:
            print(f"Different shapes: image={img_f.shape} vs dark={dimg_f.shape}")
            dark_mean = float(np.mean(dimg_f))
            dark_used = np.full_like(img_f, dark_mean)

        corrected = img_f - dark_used

        if clip_negative:
            corrected = np.maximum(corrected, 0.0)

        # Very-low threshold
        if low_clip_percentile is not None:
            if not (0.0 <= low_clip_percentile <= 100.0):
                raise ValueError("low_clip_percentile must be in [0, 100]")
            thr = float(np.percentile(corrected, low_clip_percentile))
            corrected[corrected < thr] = 0.0
        elif low_clip is not None:
            corrected[corrected < float(low_clip)] = 0.0

        # ---- DEBUG PLOT ----
        if debug_plot:
            mask_zero = corrected == 0.0

            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

            ax.imshow(corrected, cmap="gray", interpolation="none")

            overlay = np.full(corrected.shape, np.nan, dtype=float)
            overlay[mask_zero] = 1.0
            ax.imshow(overlay, cmap="Reds", vmin=0.9, vmax=1.0, interpolation="none")

            ax.set_title("Dark-subtracted image (red = zeroed pixels)")
            ax.axis("off")

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            plt.close(fig)

            print(f"Zeroed pixels: {int(mask_zero.sum())} / {mask_zero.size}")

        corrected_out = corrected.astype(self.image.dtype)
        return Image.from_array(corrected_out, path=self.path)

    def crop(self, x_min: int, x_max: int, y_min: int, y_max: int):
        "Crop the image to use only the ROI"
        if self.image is None:
            raise RuntimeError("No image loaded")

        if not (0 <= y_min < y_max <= self.ny and 0 <= x_min < x_max <= self.nx):
            raise ValueError("Invalid ROI")

        cropped = self.image[y_min:y_max, x_min:x_max]

        return Image.from_array(cropped, path=self.path)

    def remove_hot_pixels(
        self,
        size_replace: int = 15,
        use_percentile: bool = True,
        p: float = 99.0,
        use_median: bool = True,
        size_median: int = 7,
        k_med: float = 8.0,
        debug_plot: bool = True
    ):
        """
        Detect hot pixels by combining:
        (A) percentile p on the original image
        (B) outliers relative to the local median on the original image

        mask_hot = mask_percentile OR mask_median_outlier

        Replace using a masked local mean (uniform_filter) on the original image.

        Plot: original / red overlay / final.
        """
        if self.image is None:
            raise RuntimeError("No image loaded")

        # odd windows
        if size_median % 2 == 0:
            raise ValueError("size_median must be odd (3,5,7,...)")

        img = self.image
        img_f = img.astype(np.float32)

        # initialize empty masks
        mask_p = np.zeros(img.shape, dtype=bool)
        mask_m = np.zeros(img.shape, dtype=bool)

        # ---- DETECT A: percentile on the original ----
        if use_percentile:
            T = np.percentile(img_f, p)
            mask_p = img_f >= T

        # ---- DETECT B: outliers relative to local median on the original ----
        if use_median:
            med = median_filter(img_f, size=size_median, mode="nearest")
            diff_med = img_f - med

            mad = np.median(np.abs(diff_med - np.median(diff_med)))
            sigma_med = 1.4826 * mad + 1e-12

            mask_m = diff_med > (k_med * sigma_med + 1e-12)

        mask_hot = np.zeros(img.shape, dtype=bool)

        if use_percentile and use_median:
            mask_hot = mask_p | mask_m
        elif use_percentile and not use_median:
            mask_hot = mask_p
        elif not use_percentile and use_median:
            mask_hot = mask_m
        else:
            raise ValueError("Enable at least one of use_percentile and use_median")

        # ---- REPLACE: local mean ignoring hot pixels ----
        valid = (~mask_hot).astype(np.float32)

        local_sum = uniform_filter(img_f * valid, size=size_replace, mode="nearest") * (size_replace**2)
        local_count = uniform_filter(valid, size=size_replace, mode="nearest") * (size_replace**2)
        local_mean = local_sum / np.maximum(local_count, 1.0)

        corrected_f = img_f.copy()
        corrected_f[mask_hot] = local_mean[mask_hot]

        # back to original dtype
        corrected = corrected_f.astype(img.dtype)
        img_final = Image.from_array(corrected, path=self.path)

        # ---- PLOT: original / overlay / final ----
        if debug_plot:
            fig, axes = plt.subplots(
                nrows=1, ncols=2,
                figsize=(18, 6),
                dpi=300,
                sharex=True, sharey=True
            )

            # 1) Original
            axes[0].imshow(img, cmap="gray", interpolation="none")
            axes[0].set_title("Original image")
            axes[0].axis("off")

            # 2) Red overlay on original
            axes[1].imshow(img, cmap="gray", interpolation="none")
            overlay = np.full(mask_hot.shape, np.nan, dtype=float)
            overlay[mask_hot] = 1.0
            axes[1].imshow(overlay, cmap="Reds", vmin=0.9, vmax=1.0, interpolation="none")
            axes[1].set_title(f"Hot pixels: percentile p={p} + median outlier")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            plt.close(fig)

            # useful diagnostics
            if use_percentile:
                print("T (percentile) =", float(T))
                print("hot pixels (percentile) =", int(mask_p.sum()))
            if use_median:
                print("hot pixels (median outlier) =", int(mask_m.sum()))

            print("hot pixels (union) =", int(mask_hot.sum()))

        return img_final

    def difference(self, other, absolute: bool = True, plot: bool = True):
        if self.image is None or other.image is None:
            raise RuntimeError("One of the images is None")

        if self.image.shape != other.image.shape:
            raise ValueError("Images must have the same shape")

        diff = self.image.astype(np.float32) - other.image.astype(np.float32)

        if absolute:
            diff = np.abs(diff)

        diff_img = Image.from_array(diff, path=self.path)

        if plot:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

            vmax = np.nanmax(diff)
            im = ax.imshow(
                diff,
                cmap="seismic",
                vmin=-vmax if not absolute else 0,
                vmax=vmax,
                interpolation="none"
            )

            ax.set_title("Absolute difference" if absolute else "Difference I_final − I_init")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
            plt.show(block=False)
            plt.pause(0.1)
            plt.close(fig)

        return diff_img

    def compute_moments(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        verbose: bool = True,
        plot_centroid: bool = False,
        zero_low_percentile: float | None = 2.0,
        zero_debug: bool = False
    ):
        """ 
        Compute 1st- and 2nd-order moments on a 2D image
        """

        if self.image is None:
            raise RuntimeError("No image loaded")

        H = self.image.astype(np.float64)

        # Handle NaN/inf
        H = np.nan_to_num(H, copy=False, nan=0.0, posinf=None, neginf=0.0)

        # ---- ZERO "VERY LOW" PIXELS (percentile) ----
        if zero_low_percentile is not None:
            print("Zeroing outer pixels")
            if not (0.0 <= zero_low_percentile <= 100.0):
                raise ValueError("zero_low_percentile must be in [0, 100]")

            thr = float(np.percentile(H, zero_low_percentile))
            mask_low = H < thr
            H[mask_low] = 0.0

            if zero_debug:
                print(
                    f"[compute_moments] zero_low_percentile={zero_low_percentile}% -> "
                    f"thr={thr:.6g}, zeroed={int(mask_low.sum())}/{mask_low.size}"
                )

        x = np.arange(self.nx, dtype=np.float64)
        y = np.arange(self.ny, dtype=np.float64)
        X, Y = np.meshgrid(x, y)

        # discrete "dA": pixel area
        dA = dx * dy

        # Total integral (denominator)
        H_sum = np.sum(H) * dA
        print(f"Total integral: {H_sum}")
        if H_sum <= 0:
            raise ValueError("Null/non-positive total integral: check ROI, background, or data.")

        # First-order moments (centroid)
        x_bar = (np.sum(H * X) * dA) / H_sum
        y_bar = (np.sum(H * Y) * dA) / H_sum

        # Second-order moments (variance + covariance)
        sigma_x2 = (np.sum(H * (X - x_bar) ** 2) * dA) / H_sum
        sigma_y2 = (np.sum(H * (Y - y_bar) ** 2) * dA) / H_sum
        sigma_xy = (np.sum(H * (X - x_bar) * (Y - y_bar)) * dA) / H_sum

        # Root mean square
        sigma_x = np.sqrt(sigma_x2)
        sigma_y = np.sqrt(sigma_y2)

        out = dict(
            H_sum=float(H_sum),
            x_bar=float(x_bar),
            y_bar=float(y_bar),
            sigma_x2=float(sigma_x2),
            sigma_y2=float(sigma_y2),
            sigma_xy=float(sigma_xy),
            sigma_x=float(sigma_x),
            sigma_y=float(sigma_y),
            dx=float(dx),
            dy=float(dy)
        )

        if verbose:
            unit = "px" if (dx == 1.0 and dy == 1.0) else "units"
            print(f"Integral (H_sum) = {out['H_sum']:.6g}")
            print(f"x̄ = {out['x_bar']:.6g} [{unit}]   ȳ = {out['y_bar']:.6g} [{unit}]")
            print(f"σx = {out['sigma_x']:.6g} [{unit}]   σy = {out['sigma_y']:.6g} [{unit}]")
            print(f"σxy = {out['sigma_xy']:.6g} [{unit}^2]")

        if plot_centroid:
            self._plot_centroid_and_ellipses(
                H, x_bar, y_bar, sigma_x, sigma_y,
                title="Intensity distribution with centroid",
                plot_3D=True
            )

        return out

    def _plot_centroid_and_ellipses(self, img, x_bar, y_bar, sigma_x, sigma_y, title: str = "", plot_3D: bool = False):

        if plot_3D:
            fig = plt.figure(figsize=(14, 6), dpi=300)

            ax1 = fig.add_subplot(1, 2, 1)
            im2d = ax1.imshow(img, cmap="gray", interpolation="none")

            ax1.plot(
                x_bar, y_bar,
                marker="+", color="red",
                markersize=15, markeredgewidth=2,
                label=fr"Centroid $(x={x_bar:.2f},\ y={y_bar:.2f})$"
            )

            for n, color in [(1, "red"), (2, "green"), (3, "orange")]:
                label = rf"${n}\sigma:\ \sigma_x={n*sigma_x:.2f},\ \sigma_y={n*sigma_y:.2f}$"

                e = Ellipse(
                    xy=(x_bar, y_bar),
                    width=2 * n * sigma_x,
                    height=2 * n * sigma_y,
                    angle=0.0,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=2,
                    linestyle="--",
                    label=label
                )
                ax1.add_patch(e)

            ax1.set_title(title + " (2D)")
            ax1.axis("off")
            ax1.legend()
            fig.colorbar(im2d, ax=ax1, fraction=0.046)

            # 3D panel
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")

            ny, nx = img.shape[:2]

            x = np.arange(nx)
            y = np.arange(ny)
            X, Y = np.meshgrid(x, y)

            surf = ax2.plot_surface(
                X, Y, img,
                cmap="viridis",
                linewidth=0,
                antialiased=True
            )

            # Centroid also in 3D (same z level interpolated in a simple way)
            xb_i = int(np.clip(np.round(x_bar), 0, nx - 1))
            yb_i = int(np.clip(np.round(y_bar), 0, ny - 1))
            z_bar = float(img[yb_i, xb_i])

            ax2.scatter(
                [x_bar], [y_bar],
                [z_bar + 0.02 * (np.nanmax(img) - np.nanmin(img))],  # small z offset
                s=120, c="red", marker="o",
                edgecolors="black", linewidths=0.6,
                depthshade=False
            )

            ax2.set_title(title + " (3D)")
            ax2.set_xlabel("x [px]")
            ax2.set_ylabel("y [px]")
            ax2.set_zlabel("Intensity")
            fig.colorbar(surf, ax=ax2, fraction=0.046)

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            plt.close(fig)

        else:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            im = ax.imshow(img, cmap="gray", interpolation="none")

            ax.plot(
                x_bar, y_bar,
                marker="+", color="red",
                markersize=15, markeredgewidth=2,
                label=fr"Centroid $(x={x_bar:.2f},\ y={y_bar:.2f})$"
            )

            for n, color in [(1, "red"), (2, "green"), (3, "orange")]:
                label = rf"${n}\sigma:\ \sigma_x={n*sigma_x:.2f},\ \sigma_y={n*sigma_y:.2f}$"

                e = Ellipse(
                    xy=(x_bar, y_bar),
                    width=2 * n * sigma_x,
                    height=2 * n * sigma_y,
                    angle=0.0,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=2,
                    linestyle="--",
                    label=label
                )
                ax.add_patch(e)

            ax.set_title(title)
            ax.axis("off")
            ax.legend()
            plt.colorbar(im, ax=ax, fraction=0.046)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            plt.close(fig)

    def iterative_centroid_method(
        self,
        k: float = 2.0,
        eps: float = 0.1,
        max_iter: int = 30,
        dx: float = 1.0,
        dy: float = 1.0,
        verbose: bool = True,
        plot_last: bool = True
    ):

        img_curr = self
        x_off = 0
        y_off = 0

        history = []
        prev_xg = None
        prev_yg = None

        for it in range(1, max_iter + 1):
            out = img_curr.compute_moments(
                dx=dx,
                dy=dy,
                verbose=False,
                plot_centroid=False,
                zero_low_percentile=None,
                zero_debug=True
            )

            # Local centroid
            x_bar = out["x_bar"]
            y_bar = out["y_bar"]

            # Local standard deviations
            sigma_x = out["sigma_x"]
            sigma_y = out["sigma_y"]

            # Define half-size for cropping
            half_w = int(np.ceil(k * sigma_x))
            half_h = int(np.ceil(k * sigma_y))

            # Global centroid coordinates (in pixels)
            xg = x_off + x_bar
            yg = y_off + y_bar

            # Error
            if prev_xg is None:
                err = np.inf
            else:
                err = float(np.hypot(xg - prev_xg, yg - prev_yg))

            history.append(
                dict(
                    iter=it,
                    xg=float(xg), yg=float(yg),
                    sigma_x=float(sigma_x), sigma_y=float(sigma_y),
                    half_w=int(half_w), half_h=int(half_h),
                    err=float(err),
                    x_off=int(x_off), y_off=int(y_off),
                    nx=int(img_curr.nx), ny=int(img_curr.ny),
                )
            )

            if verbose:
                print(
                    f"[it={it:02d}] "
                    f"xg={xg:.3f}, yg={yg:.3f}, "
                    f"sigma_x={sigma_x:.3f}, sigma_y={sigma_y:.3f}, "
                    f"half_w={half_w}, half_h={half_h}, "
                    f"err={err:.4g}"
                )

            # Convergence criterion
            if it > 1 and err <= eps:
                if verbose:
                    print(f"Convergence reached: err={err:.4g} <= eps={eps}")
                break

            # Crop centered on the global centroid
            img_next, x_min, y_min = self.crop_centered(xg, yg, half_w, half_h)

            x_off = x_min
            y_off = y_min
            img_curr = img_next

            prev_xg, prev_yg = xg, yg

        # Final moments
        out_final = img_curr.compute_moments(dx=dx, dy=dy, verbose=verbose, plot_centroid=False)

        meta = dict(
            x_off=int(x_off),
            y_off=int(y_off),
            history=history,
        )

        if plot_last:
            # Plot with 1σ,2σ,3σ ellipses on the final ROI (local coordinates)
            self._plot_centroid_and_ellipses(
                img=img_curr.image.astype(np.float64),
                x_bar=out_final["x_bar"],
                y_bar=out_final["y_bar"],
                sigma_x=out_final["sigma_x"],
                sigma_y=out_final["sigma_y"],
                title="Final ROI (iterative centroid crop)",
                plot_3D=True
            )

        return img_curr, out_final, meta

    def crop_centered(
        self,
        x_c: float,
        y_c: float,
        half_w: int,
        half_h: int
    ):
        """
        Crop centered at (x_c, y_c) with half-sizes half_w, half_h (in pixels).
        Returns: (img_crop, x_min, y_min) where x_min,y_min are offsets relative to the original image.
        """
        if self.image is None:
            raise RuntimeError("No image loaded")

        x_c_int = int(np.round(x_c))
        y_c_int = int(np.round(y_c))

        x_min = max(0, x_c_int - half_w)
        x_max = min(self.nx, x_c_int + half_w)
        y_min = max(0, y_c_int - half_h)
        y_max = min(self.ny, y_c_int + half_h)

        # Ensure valid ROI
        if x_max <= x_min + 1 or y_max <= y_min + 1:
            raise ValueError("Crop too small or center outside the image.")

        return self.crop(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max), x_min, y_min

    def calculation_beam_widths(self, moments: dict, beam_type: str = "astigmatic", pixel_size: float = 1.0, plot: bool = False):
        """
        Calculation of laser beam widths for three cases:
            - astigmatc
            - simple_astigmatic
            - stigmatic
        """

        x0 = float(moments["x_bar"])
        y0 = float(moments["y_bar"])
        sx2 = float(moments["sigma_x2"])
        sy2 = float(moments["sigma_y2"])
        sxy = float(moments["sigma_xy"])

        bt = beam_type.strip().lower()
        if bt not in {"astigmatic", "simple_astigmatic", "stigmatic"}:
            raise ValueError("Error: Unrecognized beam type!")

        case_used = None
        phi = 0.0  # default (stigmatic or cases without rotation)

        # (3) STIGMATIC (circular): sigma_x == sigma_y
        # d_{σ} = 2√2 √(σx^2 + σy^2)
        if bt == "stigmatic":
            d_c = 2.0 * np.sqrt(2) * np.sqrt(sx2 + sy2)
            d_maj, d_min = float(d_c), float(d_c)
            case_used = "stigmatic"

        # (2) SIMPLE_ASTIGMATIC
        # d_{σx'} = 2√2 √(σx^2 + σy^2 + 2|σxy|)
        # d_{σy'} = 2√2 √(σx^2 + σy^2 - 2|σxy|)
        elif bt == "simple_astigmatic":
            abs_sxy = abs(sxy)
            d_p = 2.0 * np.sqrt(2) * np.sqrt(sx2 + sy2 + 2.0 * abs_sxy)
            d_m = 2.0 * np.sqrt(2) * np.sqrt(sx2 + sy2 - 2.0 * abs_sxy)

            d_maj, d_min = float(max(d_p, d_m)), float(min(d_p, d_m))
            case_used = "simple_astigmatic"

            phi = np.sign(sxy) * (np.pi / 4)

        # (1) General ASTIGMATIC (σx != σy and σxy != 0)
        # d_{σx'} = 2√2 { (σx^2 + σy^2) + γ[(σx^2 - σy^2)^2 + 4σxy^2]^{1/2} }^{1/2}
        # d_{σy'} = 2√2 { (σx^2 + σy^2) - γ[(σx^2 - σy^2)^2 + 4σxy^2]^{1/2} }^{1/2}
        # γ = sgn(σx^2 - σy^2)
        elif bt == "astigmatic":
            abs_sxy = abs(sxy)
            gamma = np.sign(sx2 - sy2)
            rad = np.sqrt((sx2 - sy2)**2 + 4.0 * sxy**2)
            term_p = (sx2 + sy2) + gamma * rad
            term_m = (sx2 + sy2) - gamma * rad

            d_p = 2.0 * np.sqrt(2.0) * np.sqrt(term_p)
            d_m = 2.0 * np.sqrt(2.0) * np.sqrt(term_m)

            d_maj, d_min = float(max(d_p, d_m)), float(min(d_p, d_m))
            case_used = "astigmatic"

            # Ellipticity:
            epsilon = d_min / d_maj
            if epsilon >= 0.87:
                print(f"Warning: epsilon = {epsilon} >= 0.87 -> stigmatic case should be used.")

            # Orientation (slide): φ = 1/2 arctan( 2σxy / (σx^2 - σy^2) )
            phi = 0.5 * np.arctan2(2.0 * sxy, (sx2 - sy2))

            # convert to µm
            d_maj_um = d_maj * pixel_size
            d_min_um = d_min * pixel_size

            # convert to mm
            d_maj_mm = d_maj_um * 1e-3
            d_min_mm = d_min_um * 1e-3

            # FWHM from ISO beam diameters
            fwhm_maj_px = d_maj * np.sqrt(np.log(2) / 2)
            fwhm_min_px = d_min * np.sqrt(np.log(2) / 2)

            fwhm_maj_mm = d_maj_mm * np.sqrt(np.log(2) / 2)
            fwhm_min_mm = d_min_mm * np.sqrt(np.log(2) / 2)

        # --------- PRINTS ---------
        phi_deg = float(np.degrees(phi))
        print(f"[beam widths] case = {case_used}")
        print(f"    centroid: x0 = {x0:.3f} px, y0 = {y0:.3f} px")
        print(f"    d_maj = {d_maj:.6g} px = {d_maj_mm:.6g} mm")
        print(f"    d_min = {d_min:.6g} px = {d_min_mm:.6g} mm")
        print(
            f"    FWHM_maj = {fwhm_maj_px:.6g} px = {fwhm_maj_mm:.6g} mm\n"
            f"    FWHM_min = {fwhm_min_px:.6g} px = {fwhm_min_mm:.6g} mm"
        )
        if case_used != "stigmatic":
            print(f"    phi = {phi_deg:.3f} deg")

        # --------- PLOT ---------
        if plot:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            ax.imshow(self.image, cmap="gray", interpolation="none")
            ax.plot(x0, y0, marker="+", color="red", markersize=14, markeredgewidth=2)

            if case_used == "stigmatic":
                r = 0.5 * d_maj
                patch = Circle(
                    (x0, y0),
                    radius=r,
                    edgecolor="yellow",
                    facecolor="none",
                    linewidth=2,
                    linestyle="--",
                    label=rf"$d_c={d_maj:.3g}\,\mathrm{{px}}$"
                )
            else:
                patch = Ellipse(
                    (x0, y0),
                    width=d_maj,
                    height=d_min,
                    angle=phi_deg,
                    edgecolor="yellow",
                    facecolor="none",
                    linewidth=2,
                    linestyle="--",
                    label=rf"$d_{{\max}}={d_maj:.3g}\,\mathrm{{px}},\ d_{{\min}}={d_min:.3g}\,\mathrm{{px}}$"
                )

                # semi-axes
                a = 0.5 * d_maj  # semi-major
                b = 0.5 * d_min  # semi-minor

                # rotated unit vectors
                c = np.cos(np.radians(phi_deg))
                s = np.sin(np.radians(phi_deg))

                # major axis direction (u) and minor axis direction (v)
                ux, uy = c, s
                vx, vy = -s, c

                # major axis endpoints
                x1, y1 = x0 - a * ux, y0 - a * uy
                x2, y2 = x0 + a * ux, y0 + a * uy

                # minor axis endpoints
                x3, y3 = x0 - b * vx, y0 - b * vy
                x4, y4 = x0 + b * vx, y0 + b * vy

                # draw axes
                ax.plot([x1, x2], [y1, y2], linestyle="-", linewidth=2, color="cyan", label="Major axis")
                ax.plot([x3, x4], [y3, y4], linestyle="-", linewidth=2, color="lime", label="Minor axis")

            ax.add_patch(patch)
            ax.set_title(f"Beam diameters ({case_used})")
            ax.axis("off")
            ax.legend(loc="upper right", framealpha=0.9)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            plt.close(fig)

        return {
            "case": case_used,
            "d_maj": float(d_maj),
            "d_min": float(d_min),
            "phi_rad": float(phi),
            "phi_deg": float(phi_deg),
            "x0": float(x0),
            "y0": float(y0),
            "sigma_x2": float(sx2),
            "sigma_y2": float(sy2),
            "sigma_xy": float(sxy),
        }

    def plot_beam_ellipse_global(
        self,
        results: dict,
        *,
        x0_global: float,
        y0_global: float,
        show_axes: bool = True,
        title: str | None = None,
        legend_loc: str = "upper right",
    ) -> None:

        if self.image is None:
            raise RuntimeError("No image loaded")

        case_used = str(results["case"])
        d_maj = float(results["d_maj"])
        d_min = float(results["d_min"])
        phi_deg = float(results["phi_deg"])

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.imshow(self.image, cmap="gray", interpolation="none")

        ax.plot(
            x0_global, y0_global,
            marker="+", color="red",
            markersize=14, markeredgewidth=2,
            label=f"Global centroid ({x0_global:.1f}, {y0_global:.1f}) px"
        )

        if case_used == "stigmatic":
            r = 0.5 * d_maj
            patch = Circle(
                (x0_global, y0_global),
                radius=r,
                edgecolor="yellow",
                facecolor="none",
                linewidth=2,
                linestyle="--",
                label=rf"$d_c={d_maj:.3g}\,\mathrm{{px}}$"
            )
            ax.add_patch(patch)

            if show_axes:
                ax.plot([x0_global - r, x0_global + r], [y0_global, y0_global], linewidth=2, color="red", label="Diameter x")
                ax.plot([x0_global, x0_global], [y0_global - r, y0_global + r], linewidth=2, color="red", label="Diameter y")

        else:
            patch = Ellipse(
                (x0_global, y0_global),
                width=d_maj,
                height=d_min,
                angle=phi_deg,
                edgecolor="yellow",
                facecolor="none",
                linewidth=2,
                linestyle="--",
                label=rf"$d_{{\max}}={d_maj:.3g}\,\mathrm{{px}},\ d_{{\min}}={d_min:.3g}\,\mathrm{{px}}$"
            )
            ax.add_patch(patch)

            if show_axes:
                a = 0.5 * d_maj
                b = 0.5 * d_min

                c = np.cos(np.radians(phi_deg))
                s = np.sin(np.radians(phi_deg))

                # major axis
                x1, y1 = x0_global - a * c, y0_global - a * s
                x2, y2 = x0_global + a * c, y0_global + a * s

                # minor axis (perpendicular)
                x3, y3 = x0_global + b * s, y0_global - b * c
                x4, y4 = x0_global - b * s, y0_global + b * c

                ax.plot([x1, x2], [y1, y2], linewidth=2, color="cyan", label="Major axis")
                ax.plot([x3, x4], [y3, y4], linewidth=2, color="lime", label="Minor axis")

        ax.set_title(title if title is not None else f"Beam overlay on full image ({case_used})")
        ax.axis("off")
        ax.legend(loc=legend_loc, framealpha=0.9)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)

    @staticmethod
    def _gauss2d_rotated(coords, A, x0, y0, sx, sy, phi, B):
        """
        2D rotated Gaussian with offset:
        I(x,y) = B + A * exp(-0.5 * ( (x')^2/sx^2 + (y')^2/sy^2 ))
        """
        x, y = coords
        c = np.cos(phi)
        s = np.sin(phi)

        xr = c * (x - x0) + s * (y - y0)
        yr = -s * (x - x0) + c * (y - y0)

        # Gaussian computation using sx and sy (standard deviations)
        exponent = -0.5 * (((xr / sx) ** 2) + ((yr / sy) ** 2))
        g = B + A * np.exp(exponent)

        return g.ravel()

    @staticmethod
    def _cov_from_sigmas_phi(sx, sy, phi):
        """Return covariance components (sigma_x2, sigma_y2, sigma_xy) in sensor coords."""
        c = np.cos(phi)
        s = np.sin(phi)
        sx2 = sx * sx
        sy2 = sy * sy

        # Sigma = R diag(sx2, sy2) R^T
        sigma_x2 = c * c * sx2 + s * s * sy2
        sigma_y2 = s * s * sx2 + c * c * sy2
        sigma_xy = c * s * (sx2 - sy2)
        return sigma_x2, sigma_y2, sigma_xy

    def gaussian_fitting(self, initial_moments: dict, verbose: bool = False, plot_last: bool = False):

        img = self.image.astype(np.float64)

        ny, nx = img.shape[:2]

        # Grid
        y = np.arange(ny, dtype=np.float64)
        x = np.arange(nx, dtype=np.float64)
        X, Y = np.meshgrid(x, y)

        Z = img.ravel()

        # Initialization
        x0_0 = float(initial_moments["x_bar"])
        y0_0 = float(initial_moments["y_bar"])

        # Initialize sigma with RMS (moments)
        sx_0 = float(initial_moments["sigma_x"])
        sy_0 = float(initial_moments["sigma_y"])

        # Variance initialization
        sx2_0 = float(initial_moments["sigma_x2"])
        sy2_0 = float(initial_moments["sigma_y2"])
        sxy_0 = float(initial_moments["sigma_xy"])
        phi_0 = 0.5 * np.arctan2(2.0 * sxy_0, (sx2_0 - sy2_0))

        # Initial noise
        B0 = float(np.percentile(img, 5.0))
        # Initial peak
        A0 = float(np.max(img) - B0)  # Could also be computed as img(x0, y0). Would it make more sense?

        p0 = [A0, x0_0, y0_0, sx_0, sy_0, phi_0, B0]

        # Bounds: to avoid non-physical solutions
        lower = [0.0, 0.0, 0.0, 1e-3, 1e-3, -np.pi / 2, -np.inf]
        upper = [np.inf, nx - 1.0, ny - 1.0, nx, ny, np.pi / 2, np.inf]

        # ---- fit ----
        try:
            popt, pcov = curve_fit(
                self._gauss2d_rotated,
                (X, Y),
                Z,
                p0=p0,
                bounds=(lower, upper),
                maxfev=20000
            )
        except RuntimeError as e:
            raise RuntimeError(f"Gaussian fit did not converge: {e}")

        A, x0, y0, sx, sy, phi, B = popt

        # ---- build out_conv compatible with calculation_beam_widths ----
        sigma_x2_fit, sigma_y2_fit, sigma_xy_fit = self._cov_from_sigmas_phi(sx, sy, phi)

        out_conv = dict(
            x_bar=float(x0),
            y_bar=float(y0),
            sigma_x2=float(sigma_x2_fit),
            sigma_y2=float(sigma_y2_fit),
            sigma_xy=float(sigma_xy_fit),
            sigma_x=float(np.sqrt(sigma_x2_fit)),
            sigma_y=float(np.sqrt(sigma_y2_fit)),
            # extra diagnostics
            fit_params=dict(A=float(A), B=float(B), sx=float(sx), sy=float(sy), phi_rad=float(phi)),
        )

        meta = dict(
            x_off=0,
            y_off=0,
            algorithm="gaussian_fitting",
        )

        ROI_conv = self  # no additional crop

        if verbose:
            print("[gaussian_fitting] Fit parameters")
            print(f"    A = {A:.6g}, B = {B:.6g}")
            print(f"    x0 = {x0:.3f} px, y0 = {y0:.3f} px")
            print(f"    sx = {sx:.3f} px, sy = {sy:.3f} px, phi = {np.degrees(phi):.3f} deg")

        if plot_last:
            # Quick overlay: 1/e^2 ellipse equivalent (ISO: d=4σ, but here σ is RMS sensor -> d=4*sqrt(sigma_x2_fit), etc.)
            # calculation_beam_widths will be used later; this is only a fit preview
            fit_img = (self._gauss2d_rotated((X, Y), *popt)).reshape(ny, nx)

            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            ax.imshow(img, cmap="gray", interpolation="none")
            ax.contour(fit_img, levels=6)  # no fixed colors
            ax.plot(x0, y0, marker="+", color="red", markersize=14, markeredgewidth=2)
            ax.set_title("Gaussian fit contour overlay")
            ax.axis("off")
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            plt.close(fig)

            # ---- 2nd FIGURE: fitted Gaussian (2D + 3D) ----
            fig2 = plt.figure(figsize=(14, 6), dpi=300)

            # 2D: fitted gaussian image
            ax2d = fig2.add_subplot(1, 2, 1)
            im2 = ax2d.imshow(fit_img, cmap="gray", interpolation="none")
            ax2d.plot(x0, y0, marker="+", color="red", markersize=14, markeredgewidth=2)
            ax2d.set_title("Fitted Gaussian (2D)")
            ax2d.axis("off")
            plt.colorbar(im2, ax=ax2d, fraction=0.046)

            # 3D: fitted gaussian surface
            ax3d = fig2.add_subplot(1, 2, 2, projection="3d")

            # (optional) decimation to speed up 3D rendering
            step = 4  # increase to 6 or 8 if slow
            ax3d.plot_surface(
                X[::step, ::step],
                Y[::step, ::step],
                fit_img[::step, ::step],
                cmap="viridis",
                linewidth=0,
                antialiased=True
            )
            ax3d.scatter([x0], [y0], [fit_img[int(round(y0)), int(round(x0))]],
                         s=60, c="red", depthshade=False)

            ax3d.set_title("Fitted Gaussian (3D)")
            ax3d.set_xlabel("x [px]")
            ax3d.set_ylabel("y [px]")
            ax3d.set_zlabel("Intensity")

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            plt.close(fig2)

        return ROI_conv, out_conv, meta

        # Implement Gaussian filter.

        # Implement Gaussian fitting -> 2D equation, with 4 parameters. LMSQ fitting

        return None, None, None


# Dark image should be changed, using an image with the same dimensions and accounting for burned pixels as well.
dark_image_path = r"C:\Users\luccon\Desktop\Work_MTV\PhD_2024\Calcoli_Smath\Caratterizzazione Gaussian beam\LaserMicroFlu\test.bmp"
dark = Image(dark_image_path)
# dark.plot(plot_3D=True)
image_path = r"C:\Users\luccon\Desktop\Work_MTV\PhD_2024\Calcoli_Smath\Caratterizzazione Gaussian beam\LaserMicroFlu\GaussianProfilQswitch449micros_directLaser1_Potentiometer440.tiff"
img = Image(image_path)
img.plot(plot_3D=True)
img_darkcorr = img.subtract_dark(
    dark,
    apply_dark=False,
    low_clip_percentile=10.0,
    clip_negative=True,
    debug_plot=True
)
# offsets of the manual ROI crop relative to the "large" image
X_ROI0 = 700
Y_ROI0 = 530
# img_darkcorr.plot(plot_3D=True)
ROI_init = img_darkcorr.crop(x_min=X_ROI0, x_max=2300, y_min=Y_ROI0, y_max=1960)
# img_corr = img.remove_hot_pixels(size=25, k=8.0)
img_final = img_darkcorr.remove_hot_pixels(
    size_replace=19,
    use_percentile=True,
    p=99.9,
    use_median=True,
    size_median=3,
    k_med=9.0,
    debug_plot=True
)
img_final.plot()
ROI = img_final.crop(x_min=X_ROI0, x_max=2300, y_min=Y_ROI0, y_max=1960)
# x_min=700, x_max=2300, y_min=530, y_max=1960 -> Manual crop to reduce iterations
# x_min=0, x_max=2452, y_min=0, y_max=2054
ROI.plot()
ROI_final = ROI.remove_hot_pixels(
    size_replace=19,
    use_percentile=False,
    p=99.9,
    use_median=True,
    size_median=5,
    k_med=9.0,
    debug_plot=False
)

debug = False
if debug:
    ROI_init.plot()
    ROI_final.plot()
    ROI_diff = ROI_final.difference(ROI_init, absolute=True, plot=True)

algorithm_used = "gaussian_fitting"
if algorithm_used == "gaussian_fitting":

    # Estimate initial parameters
    initial_moments = ROI_final.compute_moments(
        dx=1.0,
        dy=1.0,
        verbose=True,
        plot_centroid=False,
        zero_low_percentile=None,
        zero_debug=False
    )

    ROI_conv, out_conv, meta = ROI_final.gaussian_fitting(
        initial_moments,
        verbose=True,
        plot_last=True
    )

elif algorithm_used == "iterative_centroid":
    ROI_conv, out_conv, meta = ROI_final.iterative_centroid_method(
        k=1.90,
        eps=0.01,
        max_iter=100,
        dx=1.0,
        dy=1.0,
        verbose=True,
        plot_last=True
    )

    print("ROI global offsets:", meta["x_off"], meta["y_off"])
    print("ROI global centroid:", meta["x_off"] + out_conv["x_bar"], meta["y_off"] + out_conv["y_bar"])

# Pixel size (µm)
PIXEL_SIZE_UM = 3.45
results = ROI_conv.calculation_beam_widths(out_conv, beam_type="astigmatic", pixel_size=PIXEL_SIZE_UM, plot=True)

# Global centroid relative to the full image (img or img_final)
x0_global = X_ROI0 + meta["x_off"] + results["x0"]
y0_global = Y_ROI0 + meta["y_off"] + results["y0"]

print("Full image global centroid:", x0_global, y0_global)

img_final.plot_beam_ellipse_global(
    results,
    x0_global=x0_global,
    y0_global=y0_global,
    show_axes=True,
    title="Beam ellipse overlay on initial image"
)

## Notes

# Apply the code to test laser images (see Zotero) and check whether you obtain the same results as the original images

# Check the rotation matrices -> Understand why the ellipse rotation differs between the two algorithms.

# The computed beam diameter is underestimated compared to the initial one -> One reason could be low laser energy -> The tails are buried in noise. By performing a statistical noise study, I should be able to reduce this effect.
# Main issue is noise -> Next steps: 1) Perform noise subtraction, 2) run computations with Gaussian fitting (Gaussian filter first).

## To do
# I could easily add a plot of the ellipse also on img_final
# Compute FWHM and compare with 4 mm

# I can calibrate as a function of the laser beam measurements obtained initially -> Check the lab binder.

# ROI_final.compute_moments(dx=1.0, dy=1.0, verbose=True, plot_centroid=True)

# Ideas
# I could change the cropping method. For now, I am cropping with the original image. To make it more robust, I could fit the signal with a 2D Gaussian, possibly after applying a filter, and then crop.
# Circular cropping. Give a pixel radius and crop in a circular manner (half-height square).

# I could create a filter that detects the edges and then remove burned pixels inside.

# Laser beam diameter calculation.
# Is a Gaussian filter necessary? For interference patterns -> Try with and without the filter; if the result is very similar, then I can also use it.
