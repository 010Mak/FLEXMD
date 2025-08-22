set -Eeuo pipefail

BASE="${BASE:-http://127.0.0.1:5000}"

pp() { if command -v jq >/dev/null 2>&1; then jq -r "$1"; else cat; fi; }

echo "== preflight: lammps + reaxff + qeq styles =="
python - <<'PY' >/dev/null || { echo "missing required lammps styles (reaxff and qeq). check your env."; exit 1; }
from lammps import lammps
l = lammps()
ok = (l.has_style("pair","reaxff") or l.has_style("pair","reax/c")) and \
     (l.has_style("fix","qeq/shielded") or l.has_style("fix","qeq/reaxff") or l.has_style("fix","acks2/reaxff"))
l.close()
assert ok
PY
echo "ok"

echo
echo "== health =="
curl -sS "${BASE}/health" | pp '.'

echo
echo "== reaxff methane: 1 step =="
curl -sS -X POST "${BASE}/simulate" \
  -H "content-type: application/json" \
  --data-binary @- <<'JSON' | pp '{status:.status, last: .trajectory[-1] | {time_ps, energy, first_force: .forces[0]}}'
{
  "backend": "reaxff",
  "timestep_ps": 0.00025,
  "n_steps": 1,
  "plugin_args": { "qeq_fix": "shielded" },
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
echo "== reaxff methane: 2 steps (check energy/time arrays) =="
curl -sS -X POST "${BASE}/simulate" \
  -H "content-type: application/json" \
  --data-binary @- <<'JSON' | pp '{status:.status, times:[.trajectory[].time_ps], energies:[.trajectory[].energy]}'
{
  "backend": "reaxff",
  "timestep_ps": 0.00025,
  "n_steps": 2,
  "plugin_args": { "qeq_fix": "shielded" },
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
