#!/bin/sh
set -eu

usage() { echo "usage: $0 --git-executable PATH --gitrev-tmp FILE" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --git-executable)   GIT_EXECUTABLE=$2;    shift 2 ;;
        --output)           OUTPUT=$2;        shift 2 ;;
        *) usage ;;
    esac
done
: "${GIT_EXECUTABLE:?}" "${OUTPUT:?}"

branch=$("$GIT_EXECUTABLE" rev-parse --abbrev-ref HEAD)
hash=$("$GIT_EXECUTABLE" describe --always --dirty)
date=$("$GIT_EXECUTABLE" log -1 --format=%as)

{
    printf '#define AF_APP_GIT_BRANCH "%s"\n' "$branch"
    printf '#define AF_APP_GIT_HASH "%s"\n' "$hash"
    printf '#define AF_APP_GIT_COMMIT_LAST_CHANGED "%s"\n' "$date"
} > "$OUTPUT"

echo "Git branch: $branch"
echo "Git commit hash: $hash ($date)"
