# Diabetes Glucose Tolerance Test: Numerical and Machine Learning Solutions of a Biomedical ODE System

**SBEG108 – Numerical Methods in Biomedical Engineering**
**Faculty of Engineering, Cairo University**

---

## Overview

This project investigates the Diabetes Glucose Tolerance Test (GTT) using a coupled glucose-insulin ordinary differential equation (ODE) model from Schiesser's *Differential Equation Analysis in Biomedical Science and Engineering*.

The project reproduces the textbook results and compares classical numerical methods with a modern machine-learning-based approach for solving the same biomedical dynamical system.

---

## Project Objectives

* Reproduce the reference solution presented in Chapter 2.
* Implement two numerical methods from scratch.
* Implement a Physics-Informed Neural Network (PINN).
* Compare the methods in terms of:

  * Accuracy
  * Runtime
  * Stability
  * Solution quality

---

## ODE Model

The glucose-insulin system is represented by a set of coupled ordinary differential equations describing the evolution of:

* Glucose concentration **G(t)**
* Insulin concentration **I(t)**

The simulation begins from the fasting state and models the body's response after glucose administration during a glucose tolerance test.

### Initial Conditions

```text
G(0) = 81.14
I(0) = 5.671
```

---

## Methods

### LSODA (Reference Solution)

Adaptive ODE solver used to reproduce the textbook results and provide a reference solution.

### Forward Euler Method

First-order numerical integration method implemented from scratch.

### Runge-Kutta Fourth Order (RK4)

Fourth-order numerical integration method implemented from scratch.

### Physics-Informed Neural Network (PINN)

A neural network trained using:

* ODE residual loss
* Initial condition loss

The PINN learns continuous glucose and insulin trajectories while satisfying the governing differential equations.

---

## Comparison Metrics

The methods will be evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Square Error (RMSE)
* Runtime
* Stability
* Visual agreement with the reference solution

---

## Repository Structure

```text
src/
├── lsoda/
├── euler/
├── rk4/
└── pinn/

figures/
results/
report/
presentation/
references/
```

### Folder Descriptions

| Folder       | Purpose                                 |
| ------------ | --------------------------------------- |
| src          | Source code for all methods             |
| figures      | Generated plots and visualizations      |
| results      | Numerical outputs and comparison tables |
| report       | IEEE project report                     |
| presentation | Presentation slides                     |
| references   | Supporting papers and literature        |

---

## Results

Results will be added after implementation and testing.

### Planned Outputs

* Glucose concentration vs time
* Insulin concentration vs time
* Error comparison tables
* Runtime comparison tables
* PINN training curves

---

## Extended Results

Additional material that does not fit in the 4-page report will be included here:

* Convergence studies
* Additional simulation cases
* PINN training history
* Sensitivity analysis
* Raw numerical outputs

---

## Future Work

Potential future extensions include:

* Neural ODEs
* Patient-specific parameter estimation
* Real clinical glucose datasets
* Hybrid physics-machine-learning models

---

## Team

Faculty of Engineering, Cairo University

Biomedical Engineering Department
