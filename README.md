# PINNs_For_ShallowWaterEquations
This is a repository for my Master's Thesis "Physics Informed Neural Networks (PINNs) for Hydro-morphodynamic Modelling".

The hydrodynamic processes in rivers and coastal regions are described by the infamous Shallow Water Equations, the depth averaged version of Navier Stokes Equations.
The morphodynamic processes  are described by the Exner Equation. It evaluates the change in bed level of rivers/coasts for given bed load flux rate.

These Partial Differential Equations (PDEs) are often solved numerically using Finite Difference or Finite Volume Methods.
PINNs provide an alternative approach to solve these equations via minimization of residual loss of PDEs, initial conditions and boundary conditions.
PINNs are ideally unsupervised machine learning approach but providing observed data can help boost the convergence of solution while being constrained by physical laws of conservation. 

