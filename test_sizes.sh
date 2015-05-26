#!/bin/bash


arg1=(1 2 3 4)
arg2=(2 3 4 5)
arg3=(2 4 8 16)
arg4=(2 4 6 8) #Baseline accuracy = 85.2%
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
arg15=(4 6 8 10 12 14)
arg16=(4 5 6 7 8 9 10)
arg17=(4 5 6 7 8 9 10 11)
arg18=(5 6 7 8 9 10 11)
arg19=(4 6 8 10)
arg20=(2 4 6 8 10)
arg21=(2 4 6 8)
arg22=(1)
arg23=(2)
arg24=(1 2)
arg25=(4)
arg26=(6)
arg27=(4 6)
arg28=(6 8)

#images=("prepro-tests/b-w-blue/" "prepro-tests/b-w-green/" "prepro-tests/b-w-red/" "prepro-tests/clarity/")
for ((i=1; i<29; i++))
do
var=arg$i[@]

echo 'running test '${!var}
python phow_birdid_multi.py --image_dir "../training_2014_06_17" --dsift_size ${!var}
done