# Motor Transfer Function Identification

## Project Overview

This project implements a system identification approach for DC motors using step response analysis. The system sends a step setpoint (75 RPM) to a motor, captures encoder data, and estimates the transfer function using Error-Output (EO) identification algorithms.

## Project Structure

```
Motor_Transfer_Function_Identification/
├── code/                           # Source code files
├── data_cleaned/                   # Processed and filtered data
├── data_initial/                   # Raw captured data from encoder
├── result/                         # Identified transfer functions and analysis
├── motor_img_characteristics/      # Motor specification images
└── README.md                       # This file
```

## Methodology

### 1. Data Collection
- **Input**: Step setpoint of 75 RPM to motor controller
- **Output**: Angular velocity data captured via encoder
- **Sampling**: Discrete-time measurements at regular intervals
- **Storage**: Raw data saved in `data_initial/` folder

### 2. Data Processing
- Clean and filter raw encoder data
- Remove noise and outliers
- Processed data stored in `data_cleaned/` folder

### 3. System Identification
- Apply Error-Output (EO) identification algorithm
- Estimate second-order discrete transfer function parameters
- Results saved in `result/` folder

### 4. Transfer Function Form
The identified transfer function follows the standard discrete ARX model:

```
G(z) = Y(z)/U(z) = (b₁z⁻¹ + b₂z⁻²)/(1 + a₁z⁻¹ + a₂z⁻²)
```

Where:
- `z⁻¹` is the unit delay operator
- `b₁, b₂` are numerator coefficients (defining zeros)
- `a₁, a₂` are denominator coefficients (defining poles)
- `Y(z)` is the output (motor speed)
- `U(z)` is the input (PWM command)

## Motor Dynamics Reference

### Dynamic Behavior Classification

#### Very Fast, Responsive
**Example**: Small coreless drone motor
- **Characteristics**: Quick response, low inertia, may have slight overshoot
- **Parameters**:
  - `b₁`: 0.10 - 0.25
  - `b₂`: 0.05 - 0.15
  - `a₁`: -1.6 - -1.2
  - `a₂`: 0.60 - 0.80

#### Well-Tuned, Critically Damped
**Example**: Precision servo motor
- **Characteristics**: Fastest response without overshoot, smooth operation
- **Parameters**:
  - `b₁`: 0.15 - 0.30
  - `b₂`: 0.10 - 0.20
  - `a₁`: -1.5 - -1.1
  - `a₂`: 0.50 - 0.70

#### Slow, Overdamped
**Example**: Large gearmotor with high inertia
- **Characteristics**: Slow, sluggish response, no overshoot
- **Parameters**:
  - `b₁`: 0.02 - 0.08
  - `b₂`: 0.01 - 0.05
  - `a₁`: -1.9 - -1.7
  - `a₂`: 0.85 - 0.95

#### With Significant Time Delay
**Example**: Communication lag, slow control systems
- **Characteristics**: Response pauses for several samples before rising
- **Parameters**:
  - `b₁`: 0.01 - 0.05
  - `b₂`: 0.10 - 0.30
  - `a₁`: -1.3 - -0.9
  - `a₂`: 0.40 - 0.60

#### Oscillatory, Underdamped
**Example**: Elastic coupling, poor tuning
- **Characteristics**: Fast response but with large overshoot and ringing
- **Parameters**:
  - `b₁`: 0.20 - 0.40
  - `b₂`: 0.05 - 0.15
  - `a₁`: -1.1 - -0.5
  - `a₂`: 0.10 - 0.40

### Parameter Interpretation Guide

- **Larger |b₁|**: Faster initial response to a command
- **a₂ closer to 1.0**: Slower system dynamics (poles near z=1)
- **a₁ highly negative**: Typically indicates faster, more damped response
- **Oscillation Condition**: If (a₁)² < 4(a₂), the system has complex poles and will exhibit overshoot and ringing

## Usage

1. **Setup**: Configure motor and encoder connections
2. **Data Collection**: Run step response experiment
3. **Processing**: Clean and filter collected data
4. **Identification**: Execute EO algorithm to estimate transfer function
5. **Validation**: Analyze model fit and system characteristics
6. **Control Design**: Use identified model for controller design

## Applications

- **Motor Control**: Design PID controllers based on identified model
- **System Analysis**: Understand motor dynamics and limitations
- **Performance Optimization**: Tune system parameters for desired response
- **Predictive Modeling**: Simulate motor behavior under different conditions

## Notes

- Ensure proper sampling frequency for accurate identification
- Consider noise filtering to improve identification accuracy
- Validate results against known motor specifications
- Multiple experiments may be needed for robust parameter estimation

## Future Work

- Implement closed-loop controller design based on identified model
- Compare with other identification methods (Least Squares, Subspace)
- Extend to MIMO systems for multi-motor applications
- Real-time parameter adaptation capabilities
