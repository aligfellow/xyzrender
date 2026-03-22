# Installation

## From PyPI

```bash
pip install xyzrender
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install xyzrender
```

To test without installing:

```bash
uvx xyzrender
```

## From Source

```bash
git clone https://github.com/aligfellow/xyzrender.git
cd xyzrender
pip install .
```

Or with uv:

```bash
git clone https://github.com/aligfellow/xyzrender.git
cd xyzrender
uv tool install .
```

## Optional dependencies

Some features require additional packages:

```bash
pip install 'xyzrender[crystal]'  # VASP/QE periodic structures (phonopy)
pip install 'xyzrender[smi]'      # SMILES input (rdkit)
pip install 'xyzrender[cif]'      # CIF input (ase)
pip install 'xyzrender[all]'      # everything above
```

xyzrender auto-detects resvg-py and uses it when available. Without it, CairoSVG is used as fallback (filters silently ignored in raster output).

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```bash
git clone https://github.com/aligfellow/xyzrender.git
cd xyzrender
just setup
```

Available `just` commands:

| Command | Description |
|---------|-------------|
| `just check` | Run lint + type-check + tests |
| `just lint` | Format and lint with ruff |
| `just type` | Type-check with ty |
| `just test` | Run pytest with coverage |
| `just fix` | Auto-fix lint issues |
| `just build` | Build distribution |
| `just setup` | Install all dev dependencies |
