# Managing small and negative values

It is common for biogeochemical reservoirs to both (i) be required to be non-negative, and 
(ii) approach zero (eg oxygen below the oxic layer in a sediment). This requires
some explicit management to allow the numerical ODE / DAE solvers to operate stably and efficiently.

PALEO follows the recommended best practice for Sundials CVODE and other adaptive solvers (also including those in MATLAB), 
which is to allow -ve values within the error tolerance, and set rates to zero when this happens.

This is a FAQ for CVODE, <https://computing.llnl.gov/projects/sundials/faq#cvode_negvals>, 
'... Remember that a small negative value in y returned by CVODE, with magnitude comparable to abstol or less, is equivalent to zero as far as the computation is concerned. .... '.  See [Shampine2005](@cite) for a detailed discussion.

There are three areas that need to be addressed:

1. When calculating biogeochemical reaction rates that depend on `myvar`,  use `max(myvar, 0.0)` or similar everywhere to set rates to zero for -ve values.  Linear transport processes (eg diffusion, flux transport) should transport -ve values to maintain conservation properties.
2. Set the `abstol` solver option to control errors for near-zero values of state Variables, see [DifferentialEquations solvers](@ref).  The default value will often be too high. In some cases, it may be most efficient to tolerate -ve values, in other cases, it may be most efficient to control errors using a combination of `reltol` and `abstol` so that -ve values are not generated. The easiest way to set `abstol` in PALEO is to use `abstol=1e-5*PALEOmodel.get_statevar_norm(modeldata.solver_view_all)` to set to a small fraction (here 1e-5, experimentation will be needed) of the state variables' `norm_value` attributes (these are set in the .yaml configuration file). 
3. Defend against -ve values when using plots with log scales by explicitly setting an axis lower limit eg `ylim=(1e-9, Inf)` (without this, the autoscaling will fail and possible produce strange-looking plots eg with inverted axes). See <https://docs.juliaplots.org/latest/generated/attributes_axis/>