#!/bin/bash
# runr.sh <dir> <a> <b> <outname>  -> renders frames [a,b) in parallel, concatenates (crf16, no re-encode) into out/<outname>.mp4
R=$(cd "$(dirname "$0")" && pwd); D=$1; A=$2; B=$3; O=$4
P=$(nproc); [ $P -gt 8 ] && P=8; cd $R/$D; N=$((B-A)); Q=$(( (N+P-1)/P ))
rm -f s*.mp4 l*.txt list.txt
for i in $(seq 0 $((P-1))); do s=$((A+i*Q)); e=$((A+(i+1)*Q)); [ $e -gt $B ] && e=$B; [ $s -ge $B ] && continue
  node render.mjs $s $e s$i.mp4 > l$i.txt 2>&1 & echo "file s$i.mp4" >> list.txt; done
wait
for f in l*.txt; do grep -q DONE $f || { echo "FAIL $D $f"; tail -3 $f; exit 1; }; done
mkdir -p $R/out; ffmpeg -v error -y -f concat -safe 0 -i list.txt -c copy $R/out/$O.mp4
F=$(ffprobe -v error -count_packets -select_streams v -show_entries stream=nb_read_packets -of csv=p=0 $R/out/$O.mp4)
echo "OK $O frames=$F expected=$N"
