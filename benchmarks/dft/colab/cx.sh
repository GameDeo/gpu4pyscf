#!/bin/bash
# usage: cx.sh SESSION TIMEOUT [-b] 'shell command'
export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-/root/.secrets/adc.json}   # decoded from $GOOGLE_ADC_B64 into a mode-600 file OUTSIDE the repo
S=$1; T=$2; shift 2
BG=0; if [ "$1" == "-b" ]; then BG=1; shift; fi
CMD="$1"
python3 - "$CMD" "$BG" > /tmp/_cx_$$.py <<'PY'
import sys,json
cmd,bg=sys.argv[1],sys.argv[2]
if bg=="1":
    print("import subprocess\nsubprocess.Popen(%r,shell=True,stdout=open('/content/bg.log','a'),stderr=subprocess.STDOUT,start_new_session=True)\nprint('started')"%cmd)
else:
    print("import subprocess\nr=subprocess.run(%r,shell=True,capture_output=True,text=True)\nprint(r.stdout[-6000:]);print(r.stderr[-3000:])"%cmd)
PY
colab --auth adc exec -s $S --timeout $T -f /tmp/_cx_$$.py 2>&1 | tail -${TAILN:-60}
rm -f /tmp/_cx_$$.py
