set -Eeuo pipefail

BASE="${BASE:-http://127.0.0.1:5000}"
BACKEND="${BACKEND:-auto}"         
EX="${EX:-methane}"             
N="${N:-5}"
DT_PS="${DT_PS:-0.0005}"          
TEMP_K="${TEMP_K:-298.0}"
FRICTION="${FRICTION:-1.0}"
ALLOW_ONLINE="${ALLOW_ONLINE:-false}" 

pp(){ if command -v jq >/dev/null 2>&1; then jq -r "$1"; else cat; fi; }

if curl -fsS "$BASE/health" >/dev/null 2>&1; then
  :
else
  echo "warning: $BASE/health not reachable; is the server running?" >&2
fi

if [[ "$EX" == "water" ]]; then
  ATOMS='[
    {"element":"O","position":[0.0000,0.0000,0.0000]},
    {"element":"H","position":[0.9572,0.0000,0.0000]},
    {"element":"H","position":[-0.2399872,0.927297,0.0000]}
  ]'
elif [[ "$EX" == "hydroxide" ]]; then
  ATOMS='[
    {"element":"O","position":[0.000,0.000,0.000],"properties":{"formal_charge":-1}},
    {"element":"H","position":[0.970,0.000,0.000]}
  ]'
else
  ATOMS='[
    {"element":"C","position":[0.000,0.000,0.000]},
    {"element":"H","position":[0.629,0.629,0.629]},
    {"element":"H","position":[0.629,-0.629,-0.629]},
    {"element":"H","position":[-0.629,0.629,-0.629]},
    {"element":"H","position":[-0.629,-0.629,0.629]}
  ]'
fi

if [[ "$ALLOW_ONLINE" == "true" ]]; then ALLOW=true; else ALLOW=false; fi

PAYLOAD=$(cat <<JSON
{
  "backend": "$BACKEND",
  "timestep_ps": $DT_PS,
  "n_steps": $N,
  "report_stride": 1,
  "atoms": $ATOMS,
  "include_identity": true,
  "allow_online_names": $ALLOW,
  "thermostat": "langevin",
  "defaultConditions": { "temperature": $TEMP_K },
  "friction_coeff": $FRICTION,
  "plugin_args": {}
}
JSON
)

RESP="$(curl -sS -X POST "$BASE/simulate" \
  -H "content-type: application/json" \
  --data-binary "$PAYLOAD")" || { echo "curl failed"; exit 1; }

if [[ -z "$RESP" ]]; then
  echo "empty response from server; check $BASE and logs" >&2
  exit 1
fi

STATUS=""
if command -v jq >/dev/null 2>&1; then
  STATUS="$(printf '%s' "$RESP" | jq -r '.status // empty' 2>/dev/null || true)"
fi

if [[ "$STATUS" != "success" ]]; then
  echo "=== error response ==="
  echo "$RESP" | pp '.'
  exit 1
fi

echo "=== meta ==="
echo "$RESP" | pp '.meta'

echo
echo "=== atoms (elements only) ==="
echo "$RESP" | pp '.atoms'

echo
echo "=== atom_radii_A (space-fill vdw radii) ==="
echo "$RESP" | pp '.atom_radii_A'

echo
echo "=== identity (formula, names, etc) ==="
echo "$RESP" | pp '.identity // "none"'

echo
echo "=== trajectory summary ==="
echo "$RESP" | pp '{n_frames: (.trajectory | length), steps: [.trajectory[].step], times_ps: [.trajectory[].time_ps], energies_kcalmol: [.trajectory[].energy]}'

echo
echo "=== first frame ==="
echo "$RESP" | pp '.trajectory[0]'

echo
echo "=== last frame ==="
echo "$RESP" | pp '.trajectory[-1]'

echo
echo "=== full json (everything) ==="
echo "$RESP" | pp '.'
