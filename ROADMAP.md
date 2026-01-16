# BeamProfilerPy – Roadmap

This document outlines planned improvements and future development directions for **BeamProfilerPy**.

## Planned developments

- **Statistical noise analysis and uncertainty estimation**  
  Implement a statistical treatment of background noise to quantify uncertainties on beam diameters and FWHM.

- **Improvement of the iterative centroid algorithm**  
  Refine the `iterative_centroid` method by introducing a more robust ROI selection strategy. The current cropping approach is sensitive to ROI size and may bias moment-based estimates for low-energy or noisy beams.

- **Beam propagation analysis (M² estimation)**  
  Extend the tool to process multiple beam spot images acquired at different axial positions along the propagation direction, enabling ISO 11146–compliant estimation of beam propagation parameters (e.g. M², waist location, depth of field and divergence).

- **Pseudo-color visualization for beam intensity images**  
  Introduce pseudo-color mapping of grayscale beam images to enhance the visualization of Gaussian spot structures. Grayscale intensity levels will be mapped to color gradients (linear or nonlinear), improving contrast and revealing details in low- and high-intensity regions.
