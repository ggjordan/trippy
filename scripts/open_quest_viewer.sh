#!/bin/bash
# open_quest_viewer.sh — generate double-click launchers for a self-hosted WebXR
# splat viewer bundle (scripts/quest_viewer_bootstrap.sh's output).
# Usage: scripts/open_quest_viewer.sh <bundle-dir> <name>
# Invariants (ADR-0008-supersplat.md Stage 3, item 9):
#   - Writes ONLY into $TRIPPY_OUTPUT/deliver/quest-viewer-<name>/, same shape as
#     scripts/open_supersplat.sh -- generated .command files are delivered directly
#     with scripts/deliver.sh, not the bundle directory itself.
#   - Two launchers per bundle, both double-clickable (no Terminal/flags needed):
#       OPEN_<NAME>.command       — binds 127.0.0.1 only, plain http, opens the Mac's
#                                   own browser. For previewing on this machine.
#       OPEN_<NAME>_ON_QUEST.command — binds 0.0.0.0 (this Wi-Fi network only) over
#                                   HTTPS with a locally-generated self-signed cert,
#                                   and prints the https://<lan-ip>:<port>/ URL to
#                                   type into the Quest browser. Runs in the
#                                   FOREGROUND (Ctrl-C stops it) so the server is
#                                   never left running unattended after Jordan closes
#                                   the window -- "reachable on the home network
#                                   while this window stays open" is stated on-screen.
#   - WHY https for the Quest and not for the Mac: the WebXR Device API
#     (`navigator.xr`) is only exposed in a "secure context" per the W3C spec, and
#     Chromium enforces this at the engine level regardless of what the page does.
#     127.0.0.1/localhost is exempted from that rule; a LAN IP (what the Quest sees
#     this Mac as) is not. Plain http to the LAN IP would load the viewer but hide
#     the VR entry button entirely. This is documented browser/spec behaviour, not
#     something that needed a headset to discover -- docs/QUEST.md records that it
#     is still UNVERIFIED whether the Quest Browser's XR session then runs at a
#     usable frame rate; that needs an actual headset session.
#   - The printed/served URL includes `?webgl`: the bundle's own bootstrap script
#     defaults to WebGPU (`url.searchParams.has('webgl') ? 'webgl' : 'webgpu'`,
#     confirmed by reading the built `index.html`/`index.js`) and its VR entry
#     button only appears under the WebGL renderer (the README states WebGL is
#     "required for AR/VR"). Without `?webgl` the Quest would load the viewer with
#     no way to enter VR at all -- a silent dead end, not an error.
#   - The self-signed cert is cached under $TRIPPY_OUTPUT/tools/quest-viewer/certs/
#     (shared across every bundle) and is regenerated automatically, at launch time,
#     only when the Mac's current LAN IP differs from the one it was issued for
#     (Wi-Fi networks change). Jordan accepts the browser's one-time self-signed-
#     certificate warning on the Quest each time the cert is regenerated; that
#     one-time step is documented in docs/QUEST.md, not hidden.
#   - No new Python dependency: the LAN server is stdlib `http.server` + `ssl`
#     wrapping the socket, run via `python3 -` (an inline heredoc baked into the
#     generated .command, matching scripts/open_supersplat.sh's fully self-contained
#     launcher style) -- nothing is installed or downloaded at serve time.
#   - The generated .command hardcodes absolute paths and depends on nothing from
#     the caller's shell (Finder gives it none of those).
#   - <bundle-dir> does NOT need to exist yet at generation time: launchers can be
#     (and, 2026-09-10, routinely are) generated wired to the path a queued
#     scripts/gpu_submit.sh job will populate later, so Jordan gets one artefact to
#     wait on instead of two separate delivery steps. Both generated .command files
#     check for the bundle's SOG file (`index.sog` or `index.compressed.ply`) AND
#     `index.html` at RUN time and refuse with a plain "still converting" message
#     if either is missing, rather than erroring on a broken URL.
# Related docs: docs/decisions/ADR-0008-supersplat.md Stage 3; docs/QUEST.md;
#   scripts/quest_viewer_bootstrap.sh (builds the bundle this serves);
#   scripts/open_supersplat.sh (the launcher-generation style this follows).
set -eu
cd "$(dirname "$0")/.."
# The MAIN checkout, not a worktree: the generated .command outlives the worktree it
# was made in (same reasoning as scripts/open_mac_viewer.sh / open_supersplat.sh).
REPO_ROOT=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)

# Exported env wins over .env (tests and worktrees set TRIPPY_OUTPUT explicitly).
_pre_env_out=${TRIPPY_OUTPUT:-}
if [ -f .env ]; then . ./.env; fi
TRIPPY_OUTPUT=${_pre_env_out:-${TRIPPY_OUTPUT:-}}
TRIPPY_OUTPUT=${TRIPPY_OUTPUT:-$PWD/output}

usage() { echo "usage: open_quest_viewer.sh <bundle-dir> <name>" >&2; }

BUNDLE=${1:-}
NAME=${2:-}
[ -n "$BUNDLE" ] && [ -n "$NAME" ] || { usage; exit 2; }

# <bundle-dir> may not exist yet (see header note): resolve it to an absolute path
# without requiring it to be there, so a launcher can be wired to a path a queued
# job will populate later. Whatever state it's in now, the generated launchers do
# their own SOG/index.html check at RUN time (below) and refuse cleanly if it's
# not ready -- this generator only ever warns, it never refuses.
if [ -d "$BUNDLE" ]; then
  BUNDLE_ABS=$(cd "$BUNDLE" && pwd)
else
  case "$BUNDLE" in
    /*) BUNDLE_ABS=$BUNDLE ;;
    *) BUNDLE_ABS="$PWD/$BUNDLE" ;;
  esac
fi
_bundle_ready=1
[ -f "$BUNDLE_ABS/index.html" ] || _bundle_ready=0
if [ -f "$BUNDLE_ABS/index.html" ] && [ ! -f "$BUNDLE_ABS/index.sog" ] && [ ! -f "$BUNDLE_ABS/index.compressed.ply" ]; then
  _bundle_ready=0
fi
if [ "$_bundle_ready" -eq 0 ]; then
  echo "ℹ bundle not built yet at $BUNDLE_ABS (generating launchers wired to it anyway --" >&2
  echo "  they refuse to open until scripts/quest_viewer_bootstrap.sh has run for real)" >&2
fi

NAME_UPPER=$(printf '%s' "$NAME" | tr '[:lower:]' '[:upper:]' | tr -c 'A-Z0-9' '_')
# quest-viewer's own fixed port band: 8950-8989, chosen to sit outside both
# scripts/deliver.sh's dynamic auto-launcher range (8800-8949) and SuperSplat's
# fixed 8877, so none of the three ever collide even if run at the same time.
PORT_OFFSET=$(printf '%s' "$NAME" | cksum | awk '{print $1 % 40}')
PORT=$((8950 + PORT_OFFSET))

CERT_DIR="$TRIPPY_OUTPUT/tools/quest-viewer/certs"

DELIVER_DIR="$TRIPPY_OUTPUT/deliver/quest-viewer-$NAME"
mkdir -p "$DELIVER_DIR"

PREVIEW_CMD="$DELIVER_DIR/OPEN_${NAME_UPPER}.command"
QUEST_CMD="$DELIVER_DIR/OPEN_${NAME_UPPER}_ON_QUEST.command"

# --- Mac-preview launcher: 127.0.0.1 only, plain http, opens the Mac's browser ---
cat > "$PREVIEW_CMD" <<LAUNCHER
#!/bin/bash
# Double-click me. Previews the "$NAME" splat viewer in your Mac's browser.
# Nothing leaves this machine: bound to 127.0.0.1 only.
# This mode CANNOT show the VR entry button (WebXR needs the Quest headset itself,
# not this Mac) -- use the "_ON_QUEST" launcher next to this one for that.
DIST="$BUNDLE_ABS"
PORT=$PORT
if [ ! -f "\$DIST/index.html" ] || { [ ! -f "\$DIST/index.sog" ] && [ ! -f "\$DIST/index.compressed.ply" ]; }; then
  echo "Not ready yet: \$DIST has no index.html + SOG."
  echo "If a conversion job is still running (scripts/gpu_submit.sh), wait for it to finish"
  echo "and try again -- see $REPO_ROOT/docs/QUEST.md."
  echo "(press return to close)"
  read -r _
  exit 1
fi
cd "\$DIST" || exit 1
if ! curl -s -o /dev/null "http://127.0.0.1:\$PORT/index.html"; then
  nohup python3 -m http.server "\$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
  sleep 1
fi
open "http://127.0.0.1:\$PORT/index.html"
echo "$NAME viewer is at http://127.0.0.1:\$PORT/  (you can close this window)"
LAUNCHER
chmod +x "$PREVIEW_CMD"

# --- Quest launcher: LAN + self-signed HTTPS, foreground, prints the URL ---------
cat > "$QUEST_CMD" <<LAUNCHER
#!/bin/bash
# Double-click me on the Mac, THEN open the printed https:// URL in the Quest's
# browser. Serves "$NAME" over HTTPS on your home Wi-Fi ONLY (self-signed cert,
# generated on this machine, never sent anywhere) -- reachable by any device on
# the same network while this window stays open. Press Ctrl-C (or close this
# window) to stop serving.
#
# First time on the Quest: the browser will warn about an untrusted certificate
# (it is self-signed, made by this script, not a public CA). Accept it once
# ("Advanced" -> "Proceed to <ip> (unsafe)" or similar) -- see docs/QUEST.md.
set -eu
DIST="$BUNDLE_ABS"
PORT=$PORT
CERT_DIR="$CERT_DIR"
CERT="\$CERT_DIR/quest-viewer.crt"
KEY="\$CERT_DIR/quest-viewer.key"
IP_RECORD="\$CERT_DIR/quest-viewer.ip"

if [ ! -f "\$DIST/index.html" ] || { [ ! -f "\$DIST/index.sog" ] && [ ! -f "\$DIST/index.compressed.ply" ]; }; then
  echo "Not ready yet: \$DIST has no index.html + SOG."
  echo "If a conversion job is still running (scripts/gpu_submit.sh), wait for it to finish"
  echo "and try again -- see $REPO_ROOT/docs/QUEST.md."
  echo "(press return to close)"
  read -r _
  exit 1
fi
command -v openssl >/dev/null 2>&1 || { echo "openssl not found on PATH — cannot make the HTTPS certificate."; read -r _; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found on PATH — needed to serve HTTPS."; read -r _; exit 1; }

# Local outbound IP via a UDP "connect" (no packet is actually sent; this only
# asks the OS which interface/address would be used to reach that address) --
# more robust than guessing an interface name like en0/en1.
LAN_IP=\$(python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()" 2>/dev/null || true)
if [ -z "\$LAN_IP" ]; then
  echo "Could not determine this Mac's LAN IP (are you connected to Wi-Fi/Ethernet?)."
  read -r _
  exit 1
fi

mkdir -p "\$CERT_DIR"
NEED_CERT=1
if [ -f "\$CERT" ] && [ -f "\$KEY" ] && [ -f "\$IP_RECORD" ] && [ "\$(cat "\$IP_RECORD")" = "\$LAN_IP" ]; then
  NEED_CERT=0
fi
if [ "\$NEED_CERT" -eq 1 ]; then
  echo "Generating a self-signed HTTPS certificate for \$LAN_IP (one-time per network change)..."
  openssl req -x509 -newkey rsa:2048 -sha256 -days 365 -nodes \\
    -keyout "\$KEY" -out "\$CERT" \\
    -subj "/CN=\$LAN_IP" \\
    -addext "subjectAltName=IP:\$LAN_IP,IP:127.0.0.1,DNS:localhost" >/dev/null 2>&1
  printf '%s' "\$LAN_IP" > "\$IP_RECORD"
fi

echo "================================================================================"
echo " On the Quest browser, open:   https://\$LAN_IP:\$PORT/?webgl"
echo " (the ?webgl is required -- the viewer defaults to WebGPU, and its own VR"
echo "  entry button only appears when the WebGL renderer is forced.)"
echo " Accept the certificate warning once (self-signed, made on this Mac)."
echo " Reachable on THIS home network only, while this window stays open."
echo " Press Ctrl-C to stop serving."
echo "================================================================================"

exec python3 - "\$DIST" "\$PORT" "\$CERT" "\$KEY" <<'PYEOF'
import http.server
import ssl
import sys

directory, port, certfile, keyfile = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)


httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
httpd.serve_forever()
PYEOF
LAUNCHER
chmod +x "$QUEST_CMD"

echo "wrote $PREVIEW_CMD"
echo "wrote $QUEST_CMD"
echo "  bundle: $BUNDLE_ABS"
echo "  port:   $PORT"
echo
echo "Deliver both with:"
echo "  scripts/deliver.sh \"$PREVIEW_CMD\" quest-viewer-$NAME-preview \"Mac preview (127.0.0.1) of the $NAME splat viewer\""
echo "  scripts/deliver.sh \"$QUEST_CMD\" quest-viewer-$NAME-quest \"Self-hosted WebXR viewer for the Quest (LAN + HTTPS, this network only) -- $NAME\""
