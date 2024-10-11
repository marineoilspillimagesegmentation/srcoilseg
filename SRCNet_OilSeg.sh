
#!/usr/bin/env bash
PRJROOT=${PWD}
rootpath=datasets

for exp in SRCNet_MarineOilSeg
do
    for dataset in  oilspillimages/SAR_oilspill_images #oilspillSARimages/ERS1_oilspill_images
    do
        for lr in 0.0001
	do
#:<< !
            python ${PRJROOT}/${exp}.py\
              --mode train \
              --output_dir ${PRJROOT}/checkpoints.lr.${lr}/${exp}/${dataset}/ \
              --max_epochs 180 \
              --input_dir ${rootpath}/${dataset}/train \
              --which_direction AtoB \
	      --display_freq 160 \
	      --batch_size 1 \
              --SRCNet_weight 1.0 \
              --lr ${lr}
#!
#: << !
            python ${PRJROOT}/${exp}.py \
              --mode test \
              --output_dir ${PRJROOT}/results.lr.${lr}/${exp}/${dataset} \
              --input_dir ${rootpath}/${dataset}/test \
              --checkpoint ${PRJROOT}/checkpoints.lr.${lr}/${exp}/${dataset} \
              --lr ${lr}
#!

	done
    done
done
