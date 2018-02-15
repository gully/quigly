Inference of the K2 PSF
---

In this research note we infer the K2 PSF.  

### Outline:

1. Overview- PSF photometry for Kepler
  - Needed for clusters
  - Helps for faint objects
  - Could help estimate calibration artifacts
2. Previous/similar work
  - Nick Saunders
  - Montet
  - Cody
  - Soares-Furtado
  - Padova group
3. Method-
  - PSF shared with all targets
  - Number of parameters
  - PSF Kernel representation
  - Option 1: Delta functions convolved with PSF, undersampling
  - Option 2: PSF evaluated at each star
  - Source of PSFs (catalog or source extraction)
  - PSF is pixel convolved
  - Linearizable problem
  - GPUs can accelerate solution
4. Application- PSF photometry
  - Synthetic data generation
  - Application to synthetic data
  - Application to real patch of FFI
  
