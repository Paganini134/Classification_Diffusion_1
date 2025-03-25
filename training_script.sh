export EXP_DIR=./results
export N_STEPS=1000
export RUN_NAME=run_1
export PRIOR_TYPE=f_phi_prior
export CAT_F_PHI=_cat_f_phi
export F_PHI_TYPE=f_phi_supervised  #f_phi_self_supervised
export MODEL_VERSION_DIR=card_onehot_conditional_results/${N_STEPS}steps/nn/${RUN_NAME}/${PRIOR_TYPE}${CAT_F_PHI}/${F_PHI_TYPE}
export LOSS=card_onehot_conditional
export TASK=cifar10
export TASK1=home/iotlab/Desktop/CARD-MAIN/classification/cifar_10_1
export TASK2=home/iotlab/Desktop/CARD-MAIN/classification/cifar_10_2
export TASK3=home/iotlab/Desktop/CARD-MAIN/classification/cifar_10_3
export TASK4=home/iotlab/Desktop/CARD-MAIN/classification/cifar_10_4
export TASK5=home/iotlab/Desktop/CARD-MAIN/classification/cifar_10_5
export TASK6=home/iotlab/Desktop/CARD-MAIN/classification/cifar_10_6
export TASK7=home/iotlab/Desktop/CARD-MAIN/classification/cifar_10_7
export TASK8=home/iotlab/Desktop/CARD-MAIN/classification/cifar_10_8

export N_SPLITS=1
export DEVICE_ID=0
export N_THREADS=8

#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config configs/${TASK}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK_1} --n_splits ${N_SPLITS} --test --tune_T
python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config /${TASK1}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config /${TASK1}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config /${TASK2}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config /${TASK3}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config /${TASK4}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config /${TASK5}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config /${TASK6}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config /${TASK7}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config /${TASK8}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}
