set -euo pipefail
mkdir -p demo/vendor
curl -L https://cdn.jsdelivr.net/npm/gif.js.optimized@1.0.1/dist/gif.min.js -o demo/vendor/gif.min.js
curl -L https://cdn.jsdelivr.net/npm/gif.js.optimized@1.0.1/dist/gif.worker.js -o demo/vendor/gif.worker.js
echo "saved to demo/vendor/"
