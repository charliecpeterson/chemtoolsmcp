# Formaldehyde ORCA_ESD examples

This set exercises ORCA 6.1.1 vibronic absorption, fluorescence,
phosphorescence, and resonance Raman calculations with PBE0/def2-SVP. It is a
small execution and parser reference, not a quantitative formaldehyde
benchmark.

ORCA calls the module ORCA_ESD. Its dynamics are harmonic correlation-function
dynamics rather than a nuclear-trajectory calculation.

## Run order

1. Copy `formaldehyde_esd_gs_opt_freq.inp` to
   `formaldehyde_esd_gs.inp`, then run the copy. ORCA derives artifact names
   from the input basename, so this produces the `formaldehyde_esd_gs.hess`
   and `formaldehyde_esd_gs.xyz` expected by the spectrum inputs.
2. Run the absorption, fluorescence, and resonance Raman inputs. They use the
   vertical-gradient model and include Herzberg-Teller derivatives.
3. Copy `formaldehyde_esd_t1_displaced_opt_freq.inp` to
   `formaldehyde_esd_t1.inp`, then run the copy. The exactly planar T1 start in
   `formaldehyde_esd_t1_opt_freq.inp` converges to a first-order saddle with a
   -673.64 cm-1 out-of-plane mode. The displaced input reaches a nonplanar
   minimum whose six molecular frequencies are positive.
4. Place the resulting `formaldehyde_esd_t1.hess` beside the three
   phosphorescence inputs.
5. Run all three phosphorescence inputs. `IRoot 1`, `2`, and `3` are the three
   spin-orbit-coupled sublevels; their rates must be summed.

The phosphorescence inputs use an adiabatic gap of 23336.22 cm-1, calculated
without zero-point corrections from the PBE0/def2-SVP S0 and T1 stationary
energies. Recompute `DELE` if the geometry, method, basis, or numerical setup
changes.

The fluorescence and phosphorescence inputs use a 10 cm-1 homogeneous
linewidth. This is narrow enough for the rate calculation while retaining a
plottable spectrum. The absorption example retains the 50 cm-1 default, and
the resonance Raman example uses ORCA's default 0-0 laser energy.

## Observed ORCA 6.1.1 results

- Absorption maximum: 312.07 nm; 0-0 energy: 30844.79 cm-1.
- Fluorescence maximum: 337.34 nm; rate: 1.830239e5 s-1, corresponding to a
  5.46 microsecond radiative lifetime.
- Resonance Raman strongest computed band: 2399.90 cm-1 at a 30844.79 cm-1
  laser energy.
- Phosphorescence sublevel rates: 1.247343, 0.5652320, and 48.10585 s-1. Their
  sum is 49.91843 s-1, corresponding to a 20.03 ms radiative lifetime. The
  dominant computed band is near 442.18 nm.

These lifetimes are radiative-only values within the harmonic, gas-phase
model. They do not include internal conversion, intersystem crossing,
solvent, temperature-dependent nonradiative channels, or experimental
quantum yields.

The ORCA 6.1 manual's short resonance Raman example spells the intensity
keyword `RRINTES`, which ORCA 6.1.1 rejects. The keyword table and executable
accept `RRINTENS`, used here.
