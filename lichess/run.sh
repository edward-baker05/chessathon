#!/usr/bin/env bash
# Start the zygote, wait for the agent to finish its import, then run lichess-bot.
# Extra arguments go to lichess-bot, e.g. `lichess/run.sh -u` to upgrade the account to a bot.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
bot="$here/lichess-bot"
socket="$here/zygote.sock"

if [[ ! -d "$bot/venv" ]]; then
    echo "lichess-bot is not installed; run \`make lichess-setup\` first" >&2
    exit 1
fi
if [[ -z "${LICHESS_BOT_TOKEN:-}" ]]; then
    if [[ -f "$here/token" ]]; then
        LICHESS_BOT_TOKEN=$(tr -d '[:space:]' < "$here/token")
        export LICHESS_BOT_TOKEN
    else
        echo "no token: export LICHESS_BOT_TOKEN or put it in $here/token" >&2
        exit 1
    fi
fi

rm -f "$socket"
echo "importing the agent (about a minute); zygote log in $here/zygote.log"
"$root/.venv/bin/python" "$here/zygote.py" --socket "$socket" 2> "$here/zygote.log" &
zygote=$!
trap 'kill "$zygote" 2>/dev/null; rm -f "$socket"' EXIT

until [[ -S "$socket" ]]; do
    if ! kill -0 "$zygote" 2>/dev/null; then
        echo "the zygote exited during the import:" >&2
        cat "$here/zygote.log" >&2
        exit 1
    fi
    sleep 1
done

cd "$bot"
ENGINE_ZYGOTE_SOCKET="$socket" venv/bin/python lichess-bot.py --config "$here/config.yml" "$@"
