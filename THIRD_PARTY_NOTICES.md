# Third-party implementation references

## fluids

The piecewise Mandhane boundary implementation and the digitized
Taitel--Dukler A/C/D boundary splines used by
`steady.flow_pattern_candidates` were adapted from:

- Project: `fluids`
- File: `fluids/two_phase.py`, functions `Mandhane_Gregory_Aziz_regime` and
  `Taitel_Dukler_regime`
- Commit: `652a01344477fb881f0879b7b69bb23f20deae46`
- Repository: <https://github.com/CalebBell/fluids>
- Copyright: Copyright (C) 2016-2025 Caleb Bell
- License: MIT

The MIT license permits use, modification, distribution, and sublicensing,
provided the copyright and permission notice are retained. The upstream
license text is available at
<https://github.com/CalebBell/fluids/blob/master/LICENSE.txt>.

The underlying literature correlation is:

J. M. Mandhane, G. A. Gregory, and K. Aziz (1974), “A flow pattern map for
gas-liquid flow in horizontal pipes,” *International Journal of Multiphase
Flow*, 1(4), 537-553. DOI: `10.1016/0301-9322(74)90006-8`.

The Taitel--Dukler reference is:

Y. Taitel and A. E. Dukler (1976), “A model for predicting flow regime
transitions in horizontal and near horizontal gas-liquid flow,” *AIChE
Journal*, 22(1), 47-55. DOI: `10.1002/aic.690220105`.
