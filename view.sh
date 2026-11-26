#!/usr/bin/env bash
# Start the viewer with less typing.
#
#   ./view.sh                  pick runs from runs/ (fzf multi-select, Tab to mark)
#   ./view.sh all              load every run in runs/
#   ./view.sh euroc replica    load runs whose folder name contains each word
#   ./view.sh some/dir x.ply   anything that exists as a path is passed through
#   ./view.sh office --max-splats 1000000   options after the runs go to cslam-viewer
#
# Picks a free port pair (viewer + upload) starting at 8080 and opens the browser once the server is up.
# Set NO_BROWSER=1 to skip opening it.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
RUNS_DIR=runs

list_runs() {
    find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d \
        \( -exec test -e {}/manifest.json \; -o -exec test -e {}/trajectories.npz \; \) -print | sort
}

pick_runs() {
    if command -v fzf >/dev/null && [[ -t 0 ]]; then
        list_runs | while read -r r; do printf '%s\t%s\n' "$(basename "$r")" "$(du -sh "$r" | cut -f1)"; done |
            fzf --multi --delimiter='\t' --with-nth=1,2 --prompt='runs> ' \
                --header='Tab: mark  Enter: open  (nothing marked = current line)' | cut -f1 |
            sed "s|^|$RUNS_DIR/|"
    else
        mapfile -t all < <(list_runs)
        local i=1
        for r in "${all[@]}"; do printf '  %d) %s (%s)\n' "$i" "$(basename "$r")" "$(du -sh "$r" | cut -f1)" >&2; i=$((i + 1)); done
        read -rp "Numbers to open (space separated, empty = all): " -a picks
        if [[ ${#picks[@]} -eq 0 ]]; then printf '%s\n' "${all[@]}"; fi
        for n in "${picks[@]}"; do printf '%s\n' "${all[$((n - 1))]}"; done
    fi
}

paths=()
opts=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --*) opts+=("$@"); break ;;
        all) mapfile -t -O "${#paths[@]}" paths < <(list_runs) ;;
        *)
            if [[ -e $1 ]]; then
                paths+=("$1")
            else
                mapfile -t hits < <(list_runs | grep -i -- "$1" || true)
                if [[ ${#hits[@]} -eq 0 ]]; then
                    echo "No run in $RUNS_DIR/ matches '$1'. Available:" >&2
                    list_runs | xargs -n1 basename | sed 's/^/  /' >&2
                    exit 1
                fi
                paths+=("${hits[@]}")
            fi
            ;;
    esac
    shift
done

if [[ ${#paths[@]} -eq 0 ]]; then
    mapfile -t paths < <(pick_runs)
    [[ ${#paths[@]} -eq 0 ]] && { echo "Nothing selected." >&2; exit 1; }
fi

# Free port pair, unless the user chose one.
# Checks the listen table instead of connecting, so running viewers don't log failed handshakes.
port_busy() { [[ -n $(ss -Hltn "sport = :$1") ]]; }
port=8080
if [[ " ${opts[*]-} " != *" --port "* ]]; then
    while port_busy "$port" || port_busy $((port + 1)); do port=$((port + 2)); done
    opts+=(--port "$port")
else
    for ((i = 0; i < ${#opts[@]}; i++)); do [[ ${opts[i]} == --port ]] && port=${opts[i + 1]}; done
fi

url="http://127.0.0.1:$port"
echo "Loading: ${paths[*]##*/}"
echo "Viewer:  $url"

if [[ -z ${NO_BROWSER:-} ]] && command -v xdg-open >/dev/null; then
    (for _ in $(seq 120); do curl -fs -o /dev/null "$url" && { xdg-open "$url" >/dev/null 2>&1; break; }; sleep 0.5; done) &
fi

exec uv run cslam-viewer "${paths[@]}" "${opts[@]}"
