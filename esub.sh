#!/bin/bash

DATE=$(date +%Y_%m_%d_%H%M)

OUTPUT_DIR=$(printf "%s/%s/%s" $SCRATCH "DL" $DATE)

mkdir -p $OUTPUT_DIR


for game in cart_pole; do
    for estimator in Reinforce Gpomdp SarahPg PageStormPg Svrpg StormPg PagePg Svrpg_auto; do
        output=$(printf "%s_%s_output.txt" $game $estimator)

        bsub -W 24:00 -n 2 -R "rusage[mem=4096]" -J "cart_gpomdp" -oo $OUTPUT_DIR/$output python environment.py --game $game --estimator $estimator --output $OUTPUT_DIR
    done

done


