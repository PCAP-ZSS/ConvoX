#!/usr/bin/env bash
# Start the ConvoX demo interface (builds the program first if needed).
# usage: scripts/start_ui.sh [--port 8000]
cd "$(dirname "$0")/.."

[ -d /usr/local/cuda-12.6/bin ] && export PATH="/usr/local/cuda-12.6/bin:$PATH"
[ -x build/convox ] || bash scripts/build.sh
[ -f data/images/img01_4k.png ] || python3 scripts/gen_images.py
exec python3 ui/server.py "$@"
