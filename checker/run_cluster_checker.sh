#!/bin/bash
PARTITION=asc
ACCOUNT=asc
BUILD_TIME=00:05:00
RUN_TIME=00:05:00
CONTAINER_IMAGE=docker://gitlab.cs.pub.ro:5050/asc/asc-public/cuda-labs:1.9.1
BUILD_SCRIPT_NAME=~/asctema3/checker/utils/batch_build_checker.sh
WAIT_SCRIPT_NAME=~/asctema3/checker/utils/batch_wait.sh

cd ~/asctema3/checker
sbatch --time ${BUILD_TIME} --account ${ACCOUNT} -p ${PARTITION} --gres gpu:1 --export=CONTAINER_IMAGE=${CONTAINER_IMAGE} ${BUILD_SCRIPT_NAME} | ${WAIT_SCRIPT_NAME}

