#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-http://127.0.0.1:5000}"
BASE="${BASE%/}"

MOLECULE="${MOLECULE:-methane}"      # methane | water | hydroxide
SPLIT_MODE="${SPLIT_MODE:-atom}"     # atom | element | none
VOXEL="${VOXEL:-0.15}"               # smaller = smoother, slower, bigger zip
GAP="${GAP:-0.20}"                   # seam clearance (mm)
CENTER="${CENTER:-true}"             # true|false

POST_TO_DISCORD="${POST_TO_DISCORD:-0}"  # 1 to upload the ZIP to your configured webhook
DISCORD_MESSAGE="${DISCORD_MESSAGE:-FLEXMD spacefill kit: ${MOLECULE} (${SPLIT_MODE})}"

OUT_DIR="${OUT_DIR:-.}"
OUT="${OUT:-${OUT_DIR}/spacefill_${MOLECULE}_${SPLIT_MODE}_vox${VOXEL}_gap${GAP}.zip}"

case "${MOLECULE}" in
  methane)
    ATOMS_JSON='[
      {"element":"C","position":[0.000,0.000,0.000]},
      {"element":"H","position":[0.629,0.629,0.629]},
      {"element":"H","position":[0.629,-0.629,-0.629]},
      {"element":"H","position":[-0.629,0.629,-0.629]},
      {"element":"H","position":[-0.629,-0.629,0.629]}
    ]'
    ;;
  water)
    ATOMS_JSON='[
      {"element":"O","position":[0.0000,0.0000,0.0000]},
      {"element":"H","position":[0.9572,0.0000,0.0000]},
      {"element":"H","position":[-0.2399872,0.9272970,0.0000]}
    ]'
    ;;
  hydroxide)
    ATOMS_JSON='[
      {"element":"O","position":[0.000,0.000,0.000],"properties":{"formal_charge":-1}},
      {"element":"H","position":[0.970,0.000,0.000]}
    ]'
    ;;
  *)
    echo "Unknown MOLECULE='${MOLECULE}'. Use methane|water|hydroxide." >&2
    exit 2
    ;;
esac

echo "Checking ${BASE}/health ..."
curl -sS "${BASE}/health" >/dev/null || {
  echo "Server not reachable at ${BASE}. Is it running?" >&2
  exit 1
}

POST_BOOL=false
if [[ "${POST_TO_DISCORD}" == "1" ]]; then
  POST_BOOL=true
  echo "Discord upload: ENABLED (post_to_discord=true)"
else
  echo "Discord upload: disabled (set POST_TO_DISCORD=1 to enable)"
fi

PAYLOAD=$(cat <<JSON
{
  "size": "handheld",
  "atoms": ${ATOMS_JSON},

  "include_identity": true,
  "allow_online_names": false,

  "split_mode": "${SPLIT_MODE}",
  "gap_mm": ${GAP},
  "voxel_size_mm": ${VOXEL},
  "center": ${CENTER},

  "post_to_discord": ${POST_BOOL},
  "discord_message": "${DISCORD_MESSAGE}"
}
JSON
)

echo
echo "POST ${BASE}/export/spacefill_kit -> ${OUT}"

TMP_HEADERS="$(mktemp)"
TMP_BODY="$(mktemp)"
cleanup() { rm -f "${TMP_HEADERS}" "${TMP_BODY}"; }
trap cleanup EXIT

curl -sS -D "${TMP_HEADERS}" -o "${TMP_BODY}" \
  -X POST "${BASE}/export/spacefill_kit" \
  -H "content-type: application/json" \
  --data-binary "${PAYLOAD}"

HTTP_CODE="$(awk 'NR==1{print $2}' "${TMP_HEADERS}")"
CONTENT_TYPE="$(grep -i '^content-type:' "${TMP_HEADERS}" | tail -n1 | cut -d: -f2- | tr -d '\r' | xargs || true)"

if [[ "${HTTP_CODE}" != "200" ]]; then
  echo "HTTP ${HTTP_CODE}"
  echo "Response headers:"
  cat "${TMP_HEADERS}"
  echo
  echo "Response body:"
  cat "${TMP_BODY}"
  exit 1
fi

if [[ "${CONTENT_TYPE}" != application/zip* ]]; then
  echo "Expected application/zip but got '${CONTENT_TYPE}'"
  echo "Body (first 2KB):"
  head -c 2048 "${TMP_BODY}" || true
  exit 1
fi

mkdir -p "$(dirname "${OUT}")"
mv "${TMP_BODY}" "${OUT}"

echo
echo "ZIP saved: ${OUT}"
echo "ZIP contents + manifest summary:"

python - <<'PY' "${OUT}"
import sys, zipfile, json

path = sys.argv[1]
z = zipfile.ZipFile(path)

names = z.namelist()
print("\n".join(names))

m = json.loads(z.read("manifest.json"))

print("\n--- manifest ---")
print("schema:", m.get("schema"))
print("size_preset:", m.get("size_preset"))
print("scale:", m.get("scaling"))
print("voxel:", m.get("voxel"))
print("geometry:", m.get("geometry"))
parts = (m.get("files") or {}).get("parts") or []
print("parts_count:", len(parts))
print("parts_files:", [p.get("filename") for p in parts])

split_mode = (m.get("splitting") or {}).get("split_mode")
if split_mode != "none" and len(parts) == 0:
    print("\nWARNING: split_mode != 'none' but no parts were generated.")
    print("Check that your exporter is v2/v3 and not the old 'H->C merge' version.")
PY

echo
echo "done."
