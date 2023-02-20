#!/bin/bash
singularity exec --nv $CONTAINER_IMAGE  \
~/asctema3/checker/run_local_checker.sh &> job.out
