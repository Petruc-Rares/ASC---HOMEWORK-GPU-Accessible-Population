#!/bin/bash
# TODO: add your username
username=student

# delete directory if exists
ssh ${username}@fep.grid.pub.ro "rm -rf asctema3" &> /dev/null
# copy code to remote fep
scp -prq $(dirname "`pwd`") ${username}@fep.grid.pub.ro:~/asctema3

# run the script on fep to build and run the code on the queue
ssh ${username}@fep.grid.pub.ro "~/asctema3/checker/run_cluster_checker.sh"

#copy back the result
scp -pq ${username}@fep.grid.pub.ro:~/asctema3/tests/*.out ../tests/
