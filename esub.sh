#!/bin/bash

DATE=$(date +%Y_%m_%d_%H%M)

OUTPUT_DIR=$(printf "%s/%s/%s" $SCRATCH "DL" $DATE)

mkdir -p $OUTPUT_DIR


ITERATIONS=50
TRAJECTORIES=10000


# bsub -W 24:00 -n 2 -R "rusage[mem=4096]" -J "cart_gpomdp" -oo $OUTPUT_DIR/$output python environment.py --game $game --estimator $estimator --output $OUTPUT_DIR --iter $ITERATIONS --num_traj $TRAJECTORIES



for game in cart_pole lunar_lander mountain_car continuous_mountain_car; do
    for estimator in PageStormPg; do
        for probability in $(seq 0.1 0.1 0.9); do
            output=$(printf "%s_%s_prob:%s_output.txt" $game $estimator $probability)

            echo $output
            bsub -W 24:00 -n 1 -R "rusage[mem=4096]" -J "$game" -oo $OUTPUT_DIR/$output python environment.py --game $game --estimator $estimator --output $OUTPUT_DIR --iter $ITERATIONS --num_traj $TRAJECTORIES --prob $probability
        
        done
    done

    # for estimator in PageStormPg StormPg; do
    #     for alpha in $(seq 0 0.1 1); do
    #         output=$(printf "%s_%s_alpha:%s_output.txt" $game $estimator $alpha)

    #         echo $output
    #         bsub -W 24:00 -n 1 -R "rusage[mem=4096]" -J "$game" -oo $OUTPUT_DIR/$output python environment.py --game $game --estimator $estimator --output $OUTPUT_DIR --iter $ITERATIONS --num_traj $TRAJECTORIES --alpha $alpha
        
    #     done
    # done


    # for estimator in Reinforce Gpomdp SarahPg Svrpg StormPg PagePg Svrpg_auto PageStormPg PagePg; do
    #     lr_s=( 1e-3 5e-3 1e-2 5e-2 )
    #     flr_s=( 2e-3 1e-2 2e-2 1e-1 )

    #     for ((i=0;i<${#lr_s[@]};i++))
	#     do
		
    #         lr=${lr_s[$i]} 
    #         flr=${flr_s[$i]} 
    #         output=$(printf "%s_%s_lr%sx%s:output.txt" $game $estimator $lr $flr)

    #         echo $output
    #         bsub -W 24:00 -n 1 -R "rusage[mem=4096]" -J "$game" -oo $OUTPUT_DIR/$output python environment.py --game $game --estimator $estimator --output $OUTPUT_DIR --iter $ITERATIONS --num_traj $TRAJECTORIES --lr $lr --flr $flr
    #     done
    # done

done



# for game in cart_pole lunar_lander mountain_car continuous_mountain_car; do
#     for estimator in PageStormPg PagePg; do
#         for probability in $(seq 0.1 0.1 0.9); do
#             output=$(printf "%s_%s_prob:%s_output.txt" $game $estimator $probability)

#             echo $output
#             bsub -W 24:00 -n 1 -R "rusage[mem=4096]" -J "$game" -oo $OUTPUT_DIR/$output python environment.py --game $game --estimator $estimator --output $OUTPUT_DIR --iter $ITERATIONS --num_traj $TRAJECTORIES --prob $probability
        
#         done
#     done

#     for estimator in PageStormPg StormPg; do
#         for alpha in $(seq 0 0.1 1); do
#             output=$(printf "%s_%s_alpha:%s_output.txt" $game $estimator $alpha)

#             echo $output
#             bsub -W 24:00 -n 1 -R "rusage[mem=4096]" -J "$game" -oo $OUTPUT_DIR/$output python environment.py --game $game --estimator $estimator --output $OUTPUT_DIR --iter $ITERATIONS --num_traj $TRAJECTORIES --alpha $alpha
        
#         done
#     done


#     for estimator in Reinforce Gpomdp SarahPg Svrpg StormPg PagePg Svrpg_auto PageStormPg PagePg; do
#         lr_s=( 1e-3 5e-3 1e-2 5e-2 )
#         flr_s=( 2e-3 1e-2 2e-2 1e-1 )

#         for ((i=0;i<${#lr_s[@]};i++))
# 	    do
		
#             lr=${lr_s[$i]} 
#             flr=${flr_s[$i]} 
#             output=$(printf "%s_%s_lr%sx%s:output.txt" $game $estimator $lr $flr)

#             echo $output
#             bsub -W 24:00 -n 1 -R "rusage[mem=4096]" -J "$game" -oo $OUTPUT_DIR/$output python environment.py --game $game --estimator $estimator --output $OUTPUT_DIR --iter $ITERATIONS --num_traj $TRAJECTORIES --lr $lr --flr $flr
#         done
#     done

# done
