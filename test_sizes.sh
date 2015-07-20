#!/bin/bash

arg1=(1 2 3 4)
arg2=(2 3 4 5)
arg3=(2 4 8 16)
arg4=(2 4 6 8)
arg5=(2 4 6)
arg6=(2 4 6 8 10)
arg7=(4 5 6 7 8 9)
arg8=(4 6 8)
arg9=(4 8 16 32)
arg10=(4 8 16 32 64)
arg11=(8 16 32 64)
arg12=(4 5 6 7)
arg13=(8 9 10 11)
arg14=(4 6 8 10 12)
arg15=(4 6 8 10)
arg16=(2 4 6 8 10)
arg17=(2 4 6 8)
arg18=(1)
arg19=(4)
arg20=(6)
arg21=(4 8 12 16)

for ((i=1; i<22; i+=1))
do
var=arg$i[@]

echo 'running test '${!var}
python phow_birdid_multi.py --image_dir "../remove-test" --dsift_size ${!var} --prefix "crop hsv 2 test dsift"
done