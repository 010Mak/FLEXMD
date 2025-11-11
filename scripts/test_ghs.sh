#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-http://127.0.0.1:5000}"

pp(){ if command -v jq >/dev/null 2>&1; then jq -r "$1"; else cat; fi; }

if ! curl -fsS "$BASE/health" >/dev/null 2>&1; then
  echo "warning: $BASE/health not reachable; is the server running?" >&2
fi

ATOMS=$(cat <<'JSON'
[
  {"element":"C","position":[-1.52,0.00,0.00]},
  {"element":"C","position":[ 0.00,0.00,0.00]},
  {"element":"O","position":[ 0.00,1.23,0.00]},
  {"element":"C","position":[ 1.52,0.00,0.00]},
  {"element":"H","position":[-1.52,0.00, 1.09]},
  {"element":"H","position":[-2.10,0.90,-0.30]},
  {"element":"H","position":[-2.10,-0.90,-0.30]},
  {"element":"H","position":[ 1.52,0.00,-1.09]},
  {"element":"H","position":[ 2.10,0.90, 0.30]},
  {"element":"H","position":[ 2.10,-0.90, 0.30]}
]
JSON
)

IDENTIFY_PAYLOAD=$(cat <<JSON
{
  "allow_online_names": true,
  "include_ghs": true,
  "atoms": ${ATOMS}
}
JSON
)

SIMULATE_PAYLOAD=$(cat <<JSON
{
  "backend": "smirnoff",
  "timestep_ps": 0.001,
  "n_steps": 5,
  "report_stride": 1,
  "include_identity": true,
  "allow_online_names": true,
  "include_ghs": true,
  "atoms": ${ATOMS}
}
JSON
)

echo "=== /identify (include_ghs:true) ==="
IDENT_RESP="$(curl -fsS -H 'content-type: application/json' -d "${IDENTIFY_PAYLOAD}" "${BASE}/identify")" || {
  echo "ERROR: /identify failed" >&2; exit 1; }
echo "$IDENT_RESP" | pp '.identity | {display_name, formula, inchikey, ghs_pictograms}'
if command -v jq >/dev/null 2>&1; then
  echo "GHS (identify): $(echo "$IDENT_RESP" | jq -r '.identity.ghs_pictograms.hazard_label // ""')"
fi

echo
echo "=== /simulate (include_identity:true, include_ghs:true) ==="
SIM_RESP="$(curl -fsS -H 'content-type: application/json' -d "${SIMULATE_PAYLOAD}" "${BASE}/simulate")" || {
  echo "ERROR: /simulate failed" >&2; exit 1; }
echo "$SIM_RESP" | pp '{meta: .meta, identity: .identity?}'
if command -v jq >/dev/null 2>&1; then
  echo "GHS (simulate): $(echo "$SIM_RESP" | jq -r '.identity.ghs_pictograms.hazard_label // ""')"
fi

echo
echo "done."
