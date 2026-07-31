#!/bin/bash
# Start the SAMURAI web portal.
#
#   ./run_web.sh                 # http://127.0.0.1:5000
#   SAMURAI_PORT=8080 ./run_web.sh
#   SAMURAI_BACKEND=stub ./run_web.sh   # UI only, no SAM2 (output is not real)
#
# Run it from the repository root with the SAMURAI virtualenv activated.

set -e
cd "$(dirname "$0")"

python -c "import flask" 2>/dev/null || {
  echo "Installing web dependencies..."
  pip install -r webapp/requirements-web.txt
}

echo "SAMURAI portal on http://${SAMURAI_HOST:-127.0.0.1}:${SAMURAI_PORT:-5000}"
exec python -m webapp.app
