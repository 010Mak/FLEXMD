set -Eeuo pipefail

BASE="${BASE:-http://127.0.0.1:5000}"
pp(){ if command -v jq >/dev/null 2>&1; then jq -r "$1"; else cat; fi; }

echo "== preflight: psi4 =="
python - <<'PY' >/dev/null || { echo "missing psi4"; exit 0; }
import psi4
PY
echo "ok"

echo
echo "== health =="
curl -sS "${BASE}/health" | pp '.'

echo
echo "== psi4 water hf/sto-3g: 0.5 fs, 1 step =="
curl -sS -X POST "${BASE}/simulate" \
  -H "content-type: application/json" \
  --data-binary @- <<'JSON' | pp '{status:.status, last: .trajectory[-1] | {time_ps, energy, first_force: .forces[0]}}'
{
  "backend": "psi4",
  "timestep_ps": 0.0005,
  "n_steps": 1,
  "plugin_args": {
    "method": "hf",
    "basis": "sto-3g",
    "memory": "1 GB",
    "charge": 0,
    "multiplicity": 1
  },
  "atoms": [
    {"element": "O", "position": [ 0.000000,  0.000000,  0.000000]},
    {"element": "H", "position": [ 0.957200,  0.000000,  0.000000]},
    {"element": "H", "position": [-0.239987,  0.927297,  0.000000]}
  ]
}
JSON

echo
echo "done."
