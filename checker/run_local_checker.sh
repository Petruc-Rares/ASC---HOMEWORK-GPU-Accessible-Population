#!/bin/bash

host=$(hostname)
TIMEOUT=200
FOLDERSOL=../sol
FOLDERTESTS=../tests
GPU="local"


#listK40M="hpslwn-10 hpslwn-20 hpslwn03"
#listA100="sprmcrogpu-wn140 sprmcrogpu-wn141"
#listP100="ucsc480-wn151 xl270-wn161 xl270-wn162"
if [[ $host == hpsl* ]]; then
        GPU="K40M"
        TIMEOUT=30
fi
if [[ $host == sprmcrogpu* ]]; then
        GPU="A100"
        TIMEOUT=18
fi
if [[ $host == ucsc480* ]]; then
        GPU="P100"
        TIMEOUT=15
fi
if [[ $host == xl270* ]]; then
        GPU="P100"
        TIMEOUT=15
fi


# delete previous out files
rm -f tests/*.out

# make clean
make -C $FOLDERSOL clean

# make build
make -C $FOLDERSOL

#run with timeout
echo "--------------------------------"
echo "Executing tests on ${GPU} with timeout: ${TIMEOUT}s"
timeout $TIMEOUT $FOLDERSOL/gpu_sol 789 $FOLDERTESTS/B0.in $FOLDERTESTS/B0.out \
	423 $FOLDERTESTS/E1.in $FOLDERTESTS/E1.out \
	477 $FOLDERTESTS/M1.in $FOLDERTESTS/M1.out 
	515 $FOLDERTESTS/M2.in $FOLDERTESTS/M2.out \
	438 $FOLDERTESTS/H1.in $FOLDERTESTS/H1.out
if [ $? -ne 0 ]; then
	echo "TIMEOUT"
	echo "--------------------------------"
	echo "Final score:   0/90"
else
	echo "--------------------------------"
	final_result=0

	text_diff=$(diff -w ${FOLDERTESTS}/B0.out ${FOLDERTESTS}/B0.ref | wc -l)
	if [ $text_diff -eq 0 ]; then
		echo "B0 Passed .... 10p"
		final_result=$(($final_result+10))
	else
		echo "B0 Failed ....  0p"
	fi
	text_diff=$(diff -w ${FOLDERTESTS}/E1.out ${FOLDERTESTS}/E1.ref | wc -l)
	if [ $text_diff -eq 0 ]; then
		echo "E1 Passed .... 20p"
		final_result=$(($final_result+20))
	else
		echo "E1 Failed ....  0p"
	fi
	text_diff=$(diff -w ${FOLDERTESTS}/M1.out ${FOLDERTESTS}/M1.ref | wc -l)
	if [ $text_diff -eq 0 ]; then
		echo "M1 Passed .... 20p"
		final_result=$(($final_result+20))
	else
		echo "M1 Failed ....  0p"
	fi
	text_diff=$(diff -w ${FOLDERTESTS}/M2.out ${FOLDERTESTS}/M2.ref | wc -l)
	if [ $text_diff -eq 0 ]; then
		echo "M2 Passed .... 20p"
		final_result=$(($final_result+20))
	else
		echo "M2 Failed ....  0p"
	fi
	text_diff=$(diff -w ${FOLDERTESTS}/H1.out ${FOLDERTESTS}/H1.ref | wc -l)
	if [ $text_diff -eq 0 ]; then
		echo "H1 Passed .... 20p"
		final_result=$(($final_result+20))
	else
		echo "H1 Failed ....  0p"
	fi

	echo "--------------------------------"
	echo "Final score:   ${final_result}/90"
fi

