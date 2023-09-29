# PINNs_For_ShallowWaterEquations
This is a repository for my Master's Thesis "Physics Informed Neural Networks (PINNs) for Hydro-morphodynamic Modelling".

The hydrodynamic processes in rivers and coastal regions are described by the infamous Shallow Water Equations, the depth averaged version of Navier Stokes Equations.
The morphodynamic processes  are described by the Exner Equation. It evaluates the change in bed level of rivers/coasts for given bed load flux rate.

These Partial Differential Equations (PDEs) are often solved numerically using Finite Difference or Finite Volume Methods.
PINNs provide an alternative approach to solve these equations via minimization of residual loss of PDEs, initial conditions and boundary conditions.
PINNs are ideally unsupervised machine learning approach but providing observed data can help boost the convergence of solution while being constrained by physical laws of conservation. 
The repository contains 5 different problems:

1. 1D_Steady_BedloadTransport
  - Forward PINN problem where neural network takes space and time co-ordinates and gives out bottom topography 'z' 
  - Problem statement: to solve the evolution of bottom topography with time for steady hydrodynamics condition
  - analytical solution provided in by Christophe Berthon, Stéphane Cordier, Minh H. Le, Olivier Delestre in https://arxiv.org/abs/1112.1582

2. 1D_Transcritical_Inverse_Problem_n_initial_0_01
  - Inverse PINN for a steady Transcritical problem where the manning's roughness coefficient 'n' is computed from the observed water depth and velocities
  - benchmark problem available at:  https://www.idpoisson.fr/swashes/

3. 1D_supercritical_InverseProblem_C_initial_60
  - Inverse PINN for a steady Supercritical problem where the Chezy's roughness coefficient 'C' is computed from the observed water depth and velocities
  - benchmark problem available at:  https://www.idpoisson.fr/swashes/

4. UnsteadyErodableBump_NonScaled
  - Forward PINN problem where neural network takes space and time co-ordinates and gives out water depth 'h', velocity 'u' and bottom topography 'z'
  -  Problem statement: to solve the evolution of water depth, velocity and bottom topography with time for unsteady hydrodynamics condition
  -  reference numerical solution provided by Stéphane Cordier, Minh H. Le, Tomas Morales de Luna at https://hal.science/hal-00536267v2/preview/cml.pdf

5. WetDomainDamBreak_Anchored_200x100
  - A classic dam break problem to benchmark numerical solutions for Shallow Water Equations
  - Forward PINN problem where neural network takes space and time co-ordinates and gives out water depth 'h', velocity 'u' 
