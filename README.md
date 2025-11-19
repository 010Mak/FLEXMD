# FLEXMD — Flexible Molecular Dynamics (MMD)

[![GitHub Stars](https://img.shields.io/github/stars/010Mak/FLEXMD?style=flat)](https://github.com/010Mak/FLEXMD/stargazers)
[![CI](https://img.shields.io/github/actions/workflow/status/010Mak/FLEXMD/ci.yml?label=CI)](https://github.com/010Mak/FLEXMD/actions)
[![Issues](https://img.shields.io/github/issues/010Mak/FLEXMD)](https://github.com/010Mak/FLEXMD/issues)
[![Last Commit](https://img.shields.io/github/last-commit/010Mak/FLEXMD)](https://github.com/010Mak/FLEXMD/commits/main)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20WSL2-lightgrey)](#installation-linux--wsl2)
[![API](https://img.shields.io/badge/API-REST-green)](#api)
[![OpenMM](https://img.shields.io/badge/OpenMM-8.x-blue)](https://docs.openmm.org/latest/userguide/application/01_getting_started.html)
[![OpenFF](https://img.shields.io/badge/OpenFF-SMIRNOFF-success)](https://docs.openforcefield.org/projects/toolkit/en/stable/installation.html)
[![Psi4](https://img.shields.io/badge/Psi4-1.9.x-9cf)](https://psicode.org/psi4manual/1.9.x/conda.html)
[![LAMMPS REAXFF](https://img.shields.io/badge/LAMMPS-REAXFF-important)](https://docs.lammps.org/pair_reaxff.html)

FLEXMD is a modular molecular simulation engine you can run as a **local HTTP service** or import as a **Python library**. It ships with three backend plugins:

- **Classical MD** via **OpenMM** + **OpenFF (SMIRNOFF)**  
- Single‑point/relaxation via **Psi4**  
- **Reactive MD** via **LAMMPS/ReaxFF** (optional plugin you enable with your own LAMMPS build)

A small **plugin interface** lets you choose (or auto‑select) the best backend per system size, and a compact REST API lets you drive simulations from any client (browser, Python, curl, …).

> [!NOTE]
> All **lengths** are in **Å**, **time** in **ps**, **energy** in **kcal/mol**, and **forces** in **kcal/mol/Å** (as exposed by the REST API). Units are included in the response metadata.

---

## Contents

- [Features](#features)
- [Backends at a glance](#backends-at-a-glance)
- [Requirements](#requirements)
- [Installation (Linux & WSL2)](#installation-linux--wsl2)
  - [GPU notes for WSL2](#gpu-notes-for-wsl2)
  - [Create the environment](#create-the-environment)
  - [CPU‑only (smallest) install](#cpu-only-smallest-install)
  - [GPU‑enabled installs (NVIDIA & AMD)](#gpu-enabled-installs-nvidia--amd)
  - [Verify your installation](#verify-your-installation)
  - [Optional: LAMMPS/ReaxFF via conda or source](#optional-lammpsreaxff-via-conda-or-source)
- [Run the server](#run-the-server)
- [API](#api)
  - [`GET /health`](#get-health)
  - [`GET /status`](#get-status)
  - [`GET /demo` & examples](#get-demo--examples)
  - [`POST /identify`](#post-identify)
  - [`POST /simulate`](#post-simulate)
- [Caching](#caching)
- [Configuration (env vars)](#configuration-env-vars)
- [Extending with plugins](#extending-with-plugins)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Backends**
  - `smirnoff` — OpenMM + OpenFF Toolkit; RDKit‑aided connectivity & bond pruning; constraints removed to match FLEXMD’s Verlet integrator.
  - `psi4` — HF/DFT/post‑HF energy + analytic gradients (converted to kcal/mol and kcal/mol/Å).
  - `reaxff` — optional; add a ReaxFF plugin backed by LAMMPS for reactive chemistry.
- **Auto‑selection** (`backend: "auto"`): chooses a backend from atom count (Psi4 for very small systems; classical for larger).
- **Thermostats & integrators**: Velocity‑Verlet, optional **Langevin** thermostat.
- **Identity & GHS**: optional chemical identity (names, InChIKey) and **GHS pictograms** via online lookup.
- **Caching (optional)**: JSONL cache for recent runs; keying supports user‑provided bonds or distance‑based guesses.
- **Demo UI**: simple HTML demo served under `/demo` with built‑in examples.

---

## Backends at a glance

| Backend | When to use | Notes |
|:--|:--|:--|
| **`smirnoff`** (OpenMM + OpenFF) | Small–medium organic/biomolecular systems, **non‑reactive MD** | Install via **conda‑forge**. Verify with `python -m openmm.testInstallation`. |
| **`reaxff`** (LAMMPS) | **Reactive** events (bond breaking/forming) | Your LAMMPS must be built **with REAXFF**; verify with `lmp -h` and `pair_style reaxff`. |
| **`psi4`** (Psi4) | Energy/forces for small systems | Install via **conda‑forge** or Psi4conda; set `PSI_SCRATCH`, test with `psi4 --test`. |

> [!NOTE]
> Installing **`openff-toolkit`** brings **RDKit** and **AmberTools** by default. Use `openff-toolkit-base` only if you want to supply those yourself.

---

## Requirements

- **Python 3.10+**
- **Linux** (x86_64) or **WSL2** on Windows
- **Conda or Mamba** (Miniforge/Mambaforge recommended)
- Depending on backends you plan to use:
  - **OpenMM 8.x**
  - **OpenFF Toolkit**
  - **Psi4 1.9.x**
  - **LAMMPS** with **REAXFF** package (for `reaxff` backend)

> [!TIP]
> **Mamba** is a drop‑in replacement for conda that dramatically speeds up solves. Install Miniforge (ships with mamba) or `conda install -n base -c conda-forge mamba` and then use `mamba` everywhere.

---

## Installation (Linux & WSL2)

### GPU notes for WSL2

> [!IMPORTANT]
> On **WSL2**, **do not install a Linux NVIDIA driver inside WSL**. Install the **Windows** NVIDIA driver on the host; CUDA becomes visible inside WSL automatically. If you need developer tools (`nvcc`, samples), install the **WSL‑Ubuntu CUDA toolkit** and **avoid** meta‑packages like `cuda`/`cuda-drivers` that try to install a Linux driver. *(See NVIDIA’s CUDA‑on‑WSL guide.)*

### Create the environment

<details><summary><strong>Option A (recommended): use the pinned <code>environment.yml</code></strong></summary>

```bash
# from the repo root
mamba env create -f environment.yml
mamba activate flexmd
```
</details>
<details><summary><strong>Option B: create manually (choose stacks you need)</strong></summary>

```bash
# 1) Create a clean environment
mamba create -n flexmd -c conda-forge python=3.11 flask numpy periodictable mdanalysis ase networkx sympy matplotlib

# 2) Classical MD stack (SMIRNOFF/OpenMM)
mamba install -n flexmd -c conda-forge openmm openff-toolkit

# 3) Quantum stack (Psi4)
mamba install -n flexmd -c conda-forge psi4

# 4) Optional tools used by plugins/utilities
mamba install -n flexmd -c conda-forge rdkit basis_set_exchange
```
</details>

> [!NOTE]
> **OpenFF Toolkit** (`openff-toolkit`) installs **RDKit** and **AmberTools** automatically. If you prefer to bring your own toolkits, install `openff-toolkit-base` instead and add RDKit/AmberTools yourself.

### CPU‑only (smallest) install

If you just want to try the API on CPU:

```bash
mamba create -n flexmd -c conda-forge python=3.11 flask numpy openmm openff-toolkit psi4
mamba activate flexmd
```

### GPU‑enabled installs (NVIDIA & AMD)

- **NVIDIA (CUDA)** with **conda** (`openmm` CUDA platform included):
  ```bash
  # Ask for a build matching your driver’s CUDA version
  conda install -c conda-forge openmm cuda-version=12
  python -m openmm.testInstallation
  ```

- **NVIDIA (CUDA)** with **pip** (installs CUDA platform wheels):
  ```bash
  pip install "openmm[cuda12]"
  python -m openmm.testInstallation
  ```

- **AMD (HIP/ROCm)** with **pip** (HIP platform wheels):
  ```bash
  pip install "openmm[hip6]"
  python -m openmm.testInstallation
  ```

> [!CAUTION]
> The **conda** build of OpenMM does **not** include the **HIP** platform (AMD). Use the **pip** extra `openmm[hip6]` to add HIP/ROCm support.

### Verify your installation

- **OpenMM** (confirms CPU/GPU platforms & correctness):
  ```bash
  python -m openmm.testInstallation
  ```

- **Psi4** (set scratch dir and run self‑tests):
  ```bash
  export PSI_SCRATCH="$HOME/psi4_scratch"
  mkdir -p "$PSI_SCRATCH"
  psi4 --test
  ```

- **LAMMPS** (if you plan to use ReaxFF):
  ```bash
  lmp -h | head -n 60   # check that REAXFF appears in the enabled packages
  ```

### Optional: LAMMPS/ReaxFF via conda or source

- **Conda (quickest):**
  ```bash
  mamba install -n flexmd -c conda-forge lammps
  lmp -h | grep -i reaxff || echo "ReaxFF not enabled in this build"
  ```

- **From source (custom builds):**
  ```bash
  git clone https://github.com/lammps/lammps.git
  cd lammps && mkdir build && cd build
  cmake ../cmake -D BUILD_SHARED_LIBS=on -D PKG_REAXFF=on -D PKG_PYTHON=on -D CMAKE_BUILD_TYPE=Release
  cmake --build . -j
  sudo cmake --install .         # optional system install
  # Optional Python wheel
  make install-python             # builds & installs the lammps Python module
  ```

> [!WARNING]
> **ReaxFF** usually requires **sub‑femtosecond** timesteps (e.g., 0.25 fs = 0.00025 ps). FLEXMD enforces a **maximum** ReaxFF timestep; larger values are rejected.

---

## Run the server

```bash
mamba activate flexmd   # or: conda activate flexmd
python -m api.server    # or: python api/server.py
```

By default, the app listens on **0.0.0.0:5000** (configurable).

> [!IMPORTANT]
> On shared/cloud hosts, **bind to 127.0.0.1** and place a reverse proxy or firewall in front of the service. No authentication is built in.

---

## API

All endpoints return **JSON**. Simulation units: Å (length), ps (time), kcal/mol (energy), kcal/mol/Å (force).

### `GET /health`

Liveness probe.

```bash
curl -s http://127.0.0.1:5000/health
# {"status":"ok"}
```

### `GET /status`

Server/config snapshot plus discovered backend availability.

```bash
curl -s http://127.0.0.1:5000/status | jq .
```

### `GET /demo` & examples

```bash
curl -s http://127.0.0.1:5000/demo/examples
curl -s http://127.0.0.1:5000/demo/examples/methane | jq .
```

### `POST /identify`

Infer molecular identity (e.g., name, InChIKey); optionally include **GHS** pictograms.

```json
{
  "atoms": [
    {"element": "O", "position": [0.000, 0.000, 0.000]},
    {"element": "H", "position": [0.9572, 0.000, 0.000]},
    {"element": "H", "position": [-0.2399872, 0.927297, 0.000]}
  ],
  "allow_online_names": true,
  "include_ghs": true
}
```

```bash
curl -sS http://127.0.0.1:5000/identify \
  -H 'Content-Type: application/json' \
  -d @payload.json | jq .
```

> [!CAUTION]
> Enabling `"allow_online_names"` / `"include_ghs"` may trigger **external lookups** (e.g., PubChem). For offline‑only use, leave them `false`.

### `POST /simulate`

**Fields (common):**

- `atoms`: list of `{ "element": "C", "position": [x,y,z], "properties"?: {"formal_charge"?: int} }`
- `backend`: `"smirnoff" | "reaxff" | "psi4"` (or server default)
- `timestep_ps`: float (ps), `n_steps`: int, `report_stride`: int
- `thermostat`: `"langevin"` (optional), `friction_coeff`, `defaultConditions.temperature`
- `include_thermo`, `include_identity`, `include_render_hints`
- **Cache**: `allow_cache`, `cache_policy` (`"auto"|"off"|"read"|"write"|"rw"`), `cache_scope`, `cache_method`, `cache_tags`, `canonical_minimize`
- `plugin_args`: backend‑specific

**SMIRNOFF example**

```json
{
  "backend": "smirnoff",
  "timestep_ps": 0.001,
  "n_steps": 5,
  "report_stride": 1,
  "include_thermo": true,
  "atoms": [
    {"element":"C","position":[0.0,0.0,0.0]},
    {"element":"H","position":[0.629,0.629,0.629]},
    {"element":"H","position":[0.629,-0.629,-0.629]},
    {"element":"H","position":[-0.629,0.629,-0.629]},
    {"element":"H","position":[-0.629,-0.629,0.629]}
  ]
}
```

**ReaxFF example** (tiny timestep)

```json
{
  "backend": "reaxff",
  "timestep_ps": 0.00025,
  "n_steps": 20,
  "report_stride": 5,
  "atoms": [
    {"element": "O", "position": [0.000, 0.000, 0.000]},
    {"element": "H", "position": [0.970, 0.000, 0.000]}
  ],
  "plugin_args": {
    "ffield": "./reax/ffield.reax"
  }
}
```

**Psi4 example**

```json
{
  "backend": "psi4",
  "timestep_ps": 0.0,
  "n_steps": 1,
  "atoms": [
    {"element":"O","position":[0.0,0.0,0.0]},
    {"element":"H","position":[0.9572,0.0,0.0]},
    {"element":"H","position":[-0.2399872,0.927297,0.0]}
  ],
  "plugin_args": {
    "method": "B3LYP",
    "basis": "6-31G*",
    "charge": 0,
    "multiplicity": 1
  }
}
```

**Responses** include `meta` (units, backend, steps), `trajectory` (positions/velocities/forces/energy/time_ps), optional `identity`, `render_hints`, and `cache` details.

---

## Caching

FLEXMD can **read/write** a JSONL cache under policies: `off|read|write|rw` (or `auto`). Per‑request controls (`allow_cache`, `cache_policy`, `cache_scope`, `cache_method`, `cache_tags`, `canonical_minimize`) supplement global config. Cache keys include backend, atoms, and **bonds**; when absent, FLEXMD **guesses bonds** from covalent radii for robust keying.

---

## Configuration (env vars)

See `utilities/config.py` for defaults. Common variables:

- **Server**: `RUN_HOST`, `RUN_PORT`, `DEBUG`, `SERVER_NAME`, `SERVER_LOCATION`
- **Backends & limits**: `DEFAULT_BACKEND`, `DEFAULT_TIMESTEP_PS`, `BACKENDS`, `MAX_ATOMS`, `MAX_STEPS`, `MAX_REPORT_FRAMES`, `MAX_REQUEST_BYTES`, `REAXFF_MAX_DT_PS`
- **Cache**: `CACHE_ENABLE`, `CACHE_MODE`, `CACHE_PATH`, `CACHE_BACKENDS`, `CACHE_MAX_ATOMS`
- **Webhooks**: `DISCORD_WEBHOOK_URL`, `WEBHOOK_ON_STARTUP`, `WEBHOOK_ON_SIMULATE`

```bash
export RUN_HOST=0.0.0.0
export RUN_PORT=5000
export DEFAULT_BACKEND=smirnoff

# Cache
export CACHE_ENABLE=true
export CACHE_MODE=rw
export CACHE_PATH=./cache/molecules.jsonl
export CACHE_BACKENDS=smirnoff,reaxff
export CACHE_MAX_ATOMS=128
```

---

## Extending with plugins

Implement `ForceCalculator` (see `simulation/plugin_interface.py`) and add your plugin under `plugins/*.py`, or publish as a package with an entry point:

```toml
[project.entry-points."mmdfled.plugins"]
myforce = "your_pkg.your_mod:YourPluginClass"
```

**Contract**

- Class attributes: `NAME`, `CAPABILITY` (`"classical"|"reaxff"|"psi4"|"custom"`), `MIN_ATOMS`, `MAX_ATOMS`
- Methods:
  - `is_available() -> bool`
  - `initialize(system)` (optional)
  - `compute_forces(system) -> (N,3)` in **kcal/mol/Å**
  - `compute_energy(system) -> float` in **kcal/mol**
  - Optional: `render_hints()` (e.g., bonds)

> [!TIP]
> Auto‑selection (`backend: "auto"`) uses atom counts and a configurable threshold to choose plugins.

---

## Troubleshooting

- **OpenMM GPU not detected**
  - `python -m openmm.testInstallation` and check CUDA/OpenCL/HIP platforms.
  - On WSL2, confirm **only** the Windows driver is installed (see *GPU notes for WSL2*).

- **Psi4 scratch / permissions**
  - Set `PSI_SCRATCH` to a **writable** local path; re‑run `psi4 --test`.

- **ReaxFF timestep rejected**
  - Lower `timestep_ps` (sub‑fs). Server enforces a maximum.

- **Too many frames**
  - If `n_steps/report_stride` exceeds the frame budget, the server auto‑raises `report_stride` and returns `report_stride_was_adjusted: true`.

> [!WARNING]
> FLEXMD has **no built‑in authentication**. Do not expose it to untrusted networks without a reverse proxy and firewall rules.
