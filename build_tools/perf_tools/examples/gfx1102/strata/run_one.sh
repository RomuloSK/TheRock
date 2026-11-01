#!/usr/bin/env bash
# usage: run_one.sh NAME EXE LIBDIRS PORT TAG   -> starts engine via Strata's serve.server, runs client.py, stops it (own PID only)
set -u
NAME=$1; EXE=$2; LIBS=$3; PORT=$4; TAG=$5
B=~/rocm-env/strata-bench; W=~/Documents/Codex/2026-10-31/setu/work
python3 $B/mkcfg.py $NAME-$TAG $EXE "$LIBS" $PORT >/dev/null
cd $W/strata-rootfs/opt/strata
$B/sv/bin/python -m serve.server --engine strata --config $B/$NAME-$TAG.json --port $PORT > $B/$NAME-$TAG.server.log 2>&1 &
SP=$!
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer 1234" http://127.0.0.1:$PORT/v1/models); [ "$code" = 200 ] && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -5 $B/$NAME-$TAG.server.log; exit 1; }
  sleep 10
done
echo "$NAME-$TAG ready"; grep -E "GPU 0:|HIP device|hipBLASLt|expert cache [0-9]+ slots" $B/$NAME-$TAG.engine.log | head -4 | cut -c1-170
python3 $B/client.py $PORT $B/$NAME-$TAG.results.json 2
pkill -P $SP 2>/dev/null; kill $SP 2>/dev/null; sleep 8
for p in $(ps -eo pid,comm | awk '$2 ~ /^strata/ {print $1}'); do kill $p 2>/dev/null; done; sleep 3
echo "$NAME-$TAG done"
