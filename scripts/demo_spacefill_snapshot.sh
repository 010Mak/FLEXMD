set -Eeuo pipefail
BASE="${BASE:-http://127.0.0.1:5000}"
pp(){ if command -v jq >/dev/null 2>&1; then jq -r "$1"; else cat; fi; }

curl -sS -X POST "${BASE}/simulate" \
  -H "content-type: application/json" \
  --data-binary @- <<'JSON' | pp '{meta, atoms, atom_radii_A, last_positions: .trajectory[-1].positions}'
{
  "backend": "smirnoff",
  "timestep_ps": 0.002,
  "n_steps": 5,
  "include_atoms": true,
  "include_render_hints": true,
  "fields": ["time_ps","positions","energy"],
  "atoms": [
    {"element": "C", "position": [ 0.000,  0.000,  0.000]},
    {"element": "H", "position": [ 0.629,  0.629,  0.629]},
    {"element": "H", "position": [ 0.629, -0.629, -0.629]},
    {"element": "H", "position": [-0.629,  0.629, -0.629]},
    {"element": "H", "position": [-0.629, -0.629,  0.629]}
  ],
  "plugin_args": {
    "ff_xml": "openff-2.0.0.offxml",
    "fallback_connectivity": true,
    "partial_charge_method": "zeros"
  }
}
JSON
