set -Eeuo pipefail
BASE="${BASE:-http://127.0.0.1:5000}"

resp="$(
curl -sS -X POST "$BASE/simulate" -H 'content-type: application/json' --data-binary @- <<'JSON'
{
  "backend": "reaxff",
  "timestep_ps": 0.00025,
  "n_steps": 900,
  "report_stride": 1,
  "include_thermo": true,
  "include_identity": false,
  "plugin_args": { "qeq_fix": "shielded", "auto_resize_box": true },
  "atoms": [
    { "element": "H", "position": [-0.37, 0.0, 0.0] },
    { "element": "H", "position": [ 0.37, 0.0, 0.0] },
    { "element": "O", "position": [ 1.80, 0.0, 0.0] }
  ]
}
JSON
)"

echo "$resp" | jq -r '{
  frames:(.trajectory|length),
  first:(.trajectory[0]  | {step,time_ps, U:.energy, K:.kinetic_energy_kcal_per_mol, T:.temperature_K, Etot:.total_energy_kcal_per_mol}),
  last: (.trajectory[-1] | {step,time_ps, U:.energy, K:.kinetic_energy_kcal_per_mol, T:.temperature_K, Etot:.total_energy_kcal_per_mol}),
  deltas:{
    dU:(.trajectory[-1].energy - .trajectory[0].energy),
    dK:(.trajectory[-1].kinetic_energy_kcal_per_mol - .trajectory[0].kinetic_energy_kcal_per_mol),
    dEtot:(.trajectory[-1].total_energy_kcal_per_mol - .trajectory[0].total_energy_kcal_per_mol)
  }
}'

python -c '
import sys, json, math
d = json.load(sys.stdin)
traj = d["trajectory"]

def dist(a,b): return math.dist(a,b)

HH, OH1, OH2 = [], [], []
for fr in traj:
    p = fr["positions"]
    HH.append(dist(p[0], p[1]))
    OH1.append(dist(p[2], p[0]))  # O–H1
    OH2.append(dist(p[2], p[1]))  # O–H2

def first_persistent_cross(seq, above, below, persist):
    was_above = any(v > above for v in seq[:5])
    run = 0
    idx = None
    for i, v in enumerate(seq):
        if v < below:
            run += 1
            if run >= persist and was_above:
                idx = i - persist + 1
                break
        else:
            run = 0
            if v > above:
                was_above = True
    return idx

def first_persistent_break(seq, above, persist):
    run = 0
    for i, v in enumerate(seq):
        if v > above:
            run += 1
            if run >= persist:
                return i - persist + 1
        else:
            run = 0
    return None

OH_FORMED_AT = 1.20
HH_BROKEN_AT = 1.10
PERSIST      = 5

oh1_t = first_persistent_cross(OH1, above=1.30, below=OH_FORMED_AT, persist=PERSIST)
oh2_t = first_persistent_cross(OH2, above=1.30, below=OH_FORMED_AT, persist=PERSIST)
hh_t  = first_persistent_break(HH, above=HH_BROKEN_AT, persist=PERSIST)

def fmt_time(i):
    if i is None:
        return "—"
    # estimate dt from stored times if present
    if len(traj) >= 2:
        t0 = traj[0].get("time_ps", 0.0)
        t1 = traj[1].get("time_ps", 0.0)
        dt = t1 - t0
    else:
        dt = 0.0
    t = traj[i].get("time_ps", i*dt) if i < len(traj) else (i*dt)
    return f"frame {i} (~{t:.6f} ps)"

print(f"H–H  first={HH[0]:.3f} Å last={HH[-1]:.3f} Å min={min(HH):.3f} Å max={max(HH):.3f} Å")
print(f"O–H1 first={OH1[0]:.3f} Å last={OH1[-1]:.3f} Å min={min(OH1):.3f} Å max={max(OH1):.3f} Å")
print(f"O–H2 first={OH2[0]:.3f} Å last={OH2[-1]:.3f} Å min={min(OH2):.3f} Å max={max(OH2):.3f} Å")
print("Events:")
print("  O–H1 formed:", fmt_time(oh1_t))
print("  O–H2 formed:", fmt_time(oh2_t))
print("  H–H broken :", fmt_time(hh_t))
' <<<"$resp"
