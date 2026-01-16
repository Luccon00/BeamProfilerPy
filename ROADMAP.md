# BeamProfilerPy – Roadmap

This document outlines planned improvements and future development directions for **BeamProfilerPy**.

## Planned developments

- **Statistical noise analysis and uncertainty estimation**  
  Implement a statistical treatment of background noise to quantify uncertainties on beam diameters and FWHM.

- **Improvement of the iterative centroid algorithm**  
  Refine the `iterative_centroid` method by introducing a more robust ROI selection strategy. The current cropping approach is sensitive to ROI size and may bias moment-based estimates for low-energy or noisy beams.

- **Beam propagation analysis (M² estimation)**  
  Extend the tool to process multiple beam spot images acquired at different axial positions along the propagation direction, enabling ISO 11146–compliant estimation of beam propagation parameters (e.g. M², waist location, depth of field and divergence).
