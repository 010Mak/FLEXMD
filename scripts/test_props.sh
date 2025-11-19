#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-5000}"

echo "Hitting FLEXMD /properties on http://${HOST}:${PORT} ..."
echo

# Example: water molecule (same geometry as your demo endpoint)
JSON_PAYLOAD='{
  "atoms": [
    { "element": "O", "position": [0.000, 0.000, 0.000] },
    { "element": "H", "position": [0.9572, 0.000, 0.000] },
    { "element": "H", "position": [-0.2399872, 0.927297, 0.000] }
  ],
  "allow_online_names": true,
  "include_ghs": false,
  "include_phase_diagram": true,
  "phase_diagram_samples": 200,
  "phase_diagram_margin_K": 100.0
}'

# If jq is installed, pretty-print; otherwise just dump raw JSON
if command -v jq >/dev/null 2>&1; then
  curl -sS "http://${HOST}:${PORT}/properties" \
    -H "Content-Type: application/json" \
    -d "${JSON_PAYLOAD}" | jq .
else
  curl -sS "http://${HOST}:${PORT}/properties" \
    -H "Content-Type: application/json" \
    -d "${JSON_PAYLOAD}"
  echo
fi
