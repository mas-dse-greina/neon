#!/bin/bash
# Runs process_LUNA16.sh script for all patch sizes

for arg in "20 5 40" "25 5 40" "30 5 40" "40 5 40" "45 10 60"
do

	./process_LUNA16.sh $arg

done
