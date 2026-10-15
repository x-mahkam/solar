#!/bin/bash
# run.sh <dir>  -> out/<name>_v.mp4 (faqat video, instagram sozlamalari)
set -e
R=$(cd "$(dirname "$0")" && pwd); D=$1
read _ DUR NAME < <(grep "^$D " $R/days.txt)
P=$(nproc); [ $P -gt 8 ] && P=8; [ $P -lt 2 ] && P=2
cd $R/$D; N=$(python3 -c "import math;print(math.ceil($DUR*30))"); Q=$(( (N+P-1)/P ))
rm -f seg*.mp4 log*.txt list.txt
for i in $(seq 0 $((P-1))); do s=$((i*Q)); e=$(( (i+1)*Q )); [ $e -gt $N ] && e=$N; [ $s -ge $N ] && continue
  node render.mjs $s $e seg$i.mp4 > log$i.txt 2>&1 & echo "file seg$i.mp4" >> list.txt; done
wait
for f in log[0-9]*.txt; do grep -q DONE $f || { echo "FAIL $D: $f not done"; exit 1; }; done
mkdir -p $R/out
ffmpeg -v error -y -f concat -safe 0 -i list.txt -c:v libx264 -preset medium -crf 20 -maxrate 2700k -bufsize 5400k -pix_fmt yuv420p -movflags +faststart $R/out/${NAME}_v.mp4
F=$(ffprobe -v error -count_packets -select_streams v -show_entries stream=nb_read_packets -of csv=p=0 $R/out/${NAME}_v.mp4)
echo "OK $D $NAME frames=$F expected=$N size=$(stat -c %s $R/out/${NAME}_v.mp4)"
rm -f seg*.mp4
