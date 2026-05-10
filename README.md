# 3D Two-Phase Reservoir Simulator

Python IMPES simulator for 3D two-phase oil-water reservoir flow with pressure buildup analysis.

## Features

- 3D Cartesian reservoir grid
- Two-phase oil-water flow
- IMPES method
- Corey relative permeability
- Harmonic permeability averaging
- Sparse matrix pressure solver
- Explicit saturation update
- Production well model
- Pressure buildup analysis
- Plot generation for simulation results

## Installation

```bash
pip install -r requirements.txt
```

## Run the Simulator

```bash
python examples/run_base_case.py
```

## Requirements

See `requirements.txt`.

## Repository Status

This repository contains research code for an academic reservoir simulation project.
## Run Drawdown and Buildup Test

```bash
python examples/run_welltest_drawdown_buildup.py
```

This script runs a 72-hour drawdown followed by a 55-hour shut-in buildup test and exports the pressure data to an Excel file.
