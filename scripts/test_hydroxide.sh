set -Eeuo pipefail

BASE="${BASE:-http://127.0.0.1:5000}"
pp(){ if command -v jq >/dev/null 2>&1; then jq -r "$1"; else cat; fi; }

echo "== hydroxide with smirnoff (openmm) =="
curl -sS -X POST "${BASE}/simulate" \
  -H "content-type: application/json" \
  --data-binary @- <<'JSON' | pp '{status:.status, meta:{selected_backend:.meta.selected_backend, timestep_ps:.meta.timestep_ps}, last:{time_ps:.trajectory[-1].time_ps, energy:.trajectory[-1].energy}, identity:.identity}'
{
  "backend": "smirnoff",
  "timestep_ps": 0.002,
  "n_steps": 5,
  "include_identity": true,
  "atoms": [
    {"element":"O","position":[0.0,0.0,0.0],"properties":{"formal_charge":-1}},
    {"element":"H","position":[0.97,0.0,0.0]}
  ],
  "plugin_args": {
    "ff_xml": "openff-2.0.0.offxml",
    "partial_charge_method": "zeros",
    "fallback_connectivity": true
  }
}
JSON
