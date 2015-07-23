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
arg17=(1)
arg18=(4)
arg19=(6)
arg20=(4 8 12 16)


#arg1=(-20 -10 10 20)
#arg2=(-35 -20 -10 10 20 35)
#arg3=(-45 -20 20 45)
#arg4=(-90 -45 -20 20 45 90)
#arg5=(-45 -35 -25 -10 10 25 35 45)
#arg6=(-90 -45 -35 -20 -10 10 20 35 45 90)
#arg7=(-45 -35 -25 -10 10 25 35 45)
#arg8=(-35 -25 -20 -10 10 20 25 35)
#arg9=(-25 -20 -10 10 20 25)
#arg10=(-20 -10 -5 5 10 20)
#arg11=(-10 -5 5 10)
#arg12=(-35 -20 -10 -5 5 10 20 35)
#arg13=(-35 -10 -5 5 10 35)

for ((i=1; i<21; i+=1))
do
var=arg$i[@]

echo 'running test '${!var}
python phow_crop_aug.py --image_dir "../training_2014_09_20" --dsift_size ${!var} --prefix "test rotations" --rotation -35 -20 -10 10 20 35
done

for i in {0..15}
do
python phow_crop_aug.py --image_dir "../training_2014_09_20" --dsift_size 2 4 6 8 --prefix "test rotations" --rotation -35 -20 -10 10 20 35 --sample_seed $((RANDOM*RANDOM*RANDOM))
done
