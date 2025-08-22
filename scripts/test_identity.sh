set -euo pipefail

BASE="${BASE:-http://127.0.0.1:5000}"

curl -sS -X POST "$BASE/simulate" \
  -H "content-type: application/json" \
  --data-binary @- <<'JSON' | jq '{status, identity}'
{
  "backend": "smirnoff",
  "timestep_ps": 0.002,
  "n_steps": 1,
  "include_atoms": true,
  "include_render_hints": true,
  "include_identity": true,
  "allow_online_names": true,
  "atoms": [
    {"element": "C", "position": [ 0.000,  0.000,  0.000]},
    {"element": "H", "position": [ 0.629,  0.629,  0.629]},
    {"element": "H", "position": [ 0.629, -0.629, -0.629]},
    {"element": "H", "position": [-0.629,  0.629, -0.629]},
    {"element": "H", "position": [-0.629, -0.629,  0.629]}
  ]
}
JSON
