set -Eeuo pipefail

here="$(cd "$(dirname "$0")" && pwd)"

echo "== test: reaxff dt guard =="
bash "${here}/test_reaxff_dtguard.sh" || true

echo
echo "== test: smirnoff =="
bash "${here}/test_smirnoff.sh" || true

echo
echo "== test: psi4 =="
bash "${here}/test_psi4.sh" || true

echo
echo "all requested tests done."
