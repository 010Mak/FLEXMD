set -Eeuo pipefail

BASE="${BASE:-http://127.0.0.1:5000}"
pp(){ if command -v jq >/dev/null 2>&1; then jq -r "$1"; else cat; fi; }

echo "== preflight: openmm + openff + rdkit =="
python - <<'PY' >/dev/null || { echo "missing smirnoff deps (openmm/openff/rdkit)"; exit 1; }
import openmm, openff.toolkit, rdkit
PY
echo "ok"

echo
echo "== health =="
curl -sS "${BASE}/health" | pp '.'

echo
echo "== smirnoff methane: 2 fs, 2 steps =="
curl -sS -X POST "${BASE}/simulate" \
  -H "content-type: application/json" \
  --data-binary @- <<'JSON' | pp '{status:.status, last: .trajectory[-1] | {time_ps, energy, first_force: .forces[0]}}'
{
  "backend": "smirnoff",
  "timestep_ps": 0.002,
  "n_steps": 2,
  "plugin_args": {
    "ff_xml": "openff-2.0.0.offxml",
    "fallback_connectivity": true,
    "partial_charge_method": "zeros"
  },
  "atoms": [
    {"element": "C", "position": [ 0.000,  0.000,  0.000]},
    {"element": "H", "position": [ 0.629,  0.629,  0.629]},
    {"element": "H", "position": [ 0.629, -0.629, -0.629]},
    {"element": "H", "position": [-0.629,  0.629, -0.629]},
    {"element": "H", "position": [-0.629, -0.629,  0.629]}
  ]
}
JSON

echo
echo "== smirnoff methane: 2 fs, 10 steps (times/energies) =="
curl -sS -X POST "${BASE}/simulate" \
  -H "content-type: application/json" \
  --data-binary @- <<'JSON' | pp '{status:.status, times:[.trajectory[].time_ps], energies:[.trajectory[].energy]}'
{
  "backend": "smirnoff",
  "timestep_ps": 0.002,
  "n_steps": 10,
  "plugin_args": {
    "ff_xml": "openff-2.0.0.offxml",
    "fallback_connectivity": true,
    "partial_charge_method": "zeros"
  },
  "atoms": [
    {"element": "C", "position": [ 0.000,  0.000,  0.000]},
    {"element": "H", "position": [ 0.629,  0.629,  0.629]},
    {"element": "H", "position": [ 0.629, -0.629, -0.629]},
    {"element": "H", "position": [-0.629,  0.629, -0.629]},
    {"element": "H", "position": [-0.629, -0.629,  0.629]}
  ]
}
JSON

echo
echo "done."
