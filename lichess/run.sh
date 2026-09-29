#!/usr/bin/env bash
# Build the C++ engine if it is out of date, then run lichess-bot with it.
# Extra arguments go to lichess-bot, e.g. `lichess/run.sh -u` to upgrade the account to a bot.
#
# lichess-bot starts the engine binary itself, once per game. It starts in milliseconds, so
# nothing has to be kept warm; zygote.py remains for serving Python builds to tools/sprt.py.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
bot="$here/lichess-bot"

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

make -C "$root/cpp" --no-print-directory
# A broken build should not reach a rated game: the move generator has to pass first.
make -C "$root/cpp" --no-print-directory perft

cd "$bot"
venv/bin/python lichess-bot.py --config "$here/config.yml" "$@"
