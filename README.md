# **BeamProfilerPy**
🔬 **Python-based laser beam profiling tool compliant with ISO 11146**

BeamProfilerPy is a Python script for **laser beam characterization from intensity images**, designed for experimental laboratory use.
It provides a complete workflow for **beam centroid estimation, beam width and FWHM computation**, and **2D/3D visualization**, following **ISO 11146** recommendations.

The tool is particularly suited for **Gaussian and astigmatic beams** acquired with camera-based diagnostics.

---

## ✨ Features

BeamProfilerPy includes:

### 🔹 Image preprocessing
- Dark-frame subtraction
- Robust hot-pixel detection and correction  
  (percentile-based + local median outlier detection)
- Median filtering for noise suppression
- Gaussian filtering for smoothing interference patterns

### 🔹 Beam parameter estimation (ISO 11146)
Two independent algorithms are implemented:

1. **Moment-based iterative centroid method**  
   - Based on 1st- and 2nd-order spatial moments  
   - Iterative ROI refinement  
   - Fully compliant with ISO 11146 centroid definition  

2. **Rotated 2D Gaussian fitting**  
   - Least-squares fit of a rotated elliptical Gaussian  
   - Automatic estimation of centroid, principal axes, and orientation

### 🔹 Beam width and FWHM computation
- Stigmatic, simple astigmatic, and general astigmatic cases
- Major/minor beam diameters
- FWHM computation from ISO beam diameters
- Automatic ellipse orientation evaluation

### 🔹 Visualization tools
- 2D intensity maps
- 3D intensity surfaces
- Beam ellipses overlaid on ROI or full image
- Diagnostic plots for centroid convergence and Gaussian fitting

---

## 🛠 Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- imageio

Install dependencies with:

```bash
pip install numpy scipy matplotlib imageio
```

---

## 🚀 Usage overview

Typical workflow:
1. Load the image
2. Apply preprocessing
3. Define a region of interest (ROI)
4. Choose the estimation algorithm
5. Compute beam widths and FWHM
6. Visualize results

---

## 📐 Standards

- Beam characterization follows **ISO 11146**
- Beam diameters computed from second-order moments
- FWHM derived consistently from ISO beam widths

---

## 📜 License

Copyright (c) 2026  
**Institut Clément Ader**

Released under the **MIT License**.

---

## 👤 Author

**Andrea Luccon**  
Institut Clément Ader
