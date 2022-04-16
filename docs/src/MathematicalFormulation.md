# Mathematical formulation of the reaction-transport problem

The PALEO models define various special cases of a general DAE problem (these can be combined, providing the number 
of implicit state variables ``S_{impl}`` is equal to the number of algebraic constraints ``G`` plus the number of total variables ``U``):

## Explicit ODE
The time derivative of explicit state variables ``S_{explicit}`` (a subset of all state variables ``S_{all}``) are an explicit function of time ``t``:
```math
\frac{dS_{explicit}}{dt} = F(S_{all}, t)
```
where explicit state variables ``S_{explicit}`` are identified by PALEO attribute `:vfunction = PALEOboxes.VF_StateExplicit` and paired time derivatives ``F`` by `:vfunction = PALEOboxes.VF_Deriv` along with the naming convention `<statevarname>, <statevarname>_sms`.

## Algebraic constraints
State variables ``S_{impl}`` (a subset of all state variables ``S_{all}``) are defined by algebraic constraints ``G``:
```math
0 = G(S_{all}, t)
```
where implicit state variables ``S_{impl}`` are identified by PALEO attribute `:vfunction = PALEOboxes.VF_State` and algebraic constaints ``G`` by `:vfunction = PALEOboxes.VF_Constraint` (these are not paired).

## ODE with variable substitution
State variables ``S_{impl}`` (a subset of all state variables ``S_{all}``) are defined the time evolution of total variables ``U(S_{all})`` (this case is common in biogeochemistry where the total variables ``U`` represent conserved chemical elements, and the state variables eg primary species):
```math
\frac{dU(S_{all})}{dt} = F(U(S_{all}), t)
```
where total variables ``U`` are identified by PALEO attribute `:vfunction = PALEOboxes.VF_Total` and paired time derivatives ``F`` by `:vfunction = PALEOboxes.VF_Deriv` along with the naming convention `<totalvarname>, <totalvarname>_sms`, and implicit state variables ``S_{impl}`` are identified by PALEO attribute `:vfunction = PALEOboxes.VF_State`.



