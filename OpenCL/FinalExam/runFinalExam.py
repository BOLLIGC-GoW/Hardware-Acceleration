#!/usr/bin/env python

import sys
import os, subprocess, re

#Global dictionary that holds timing data
timing_data = []

def runtest(dimension, filter_size):
	os.system("make clean")


	#I tried to pass in variable to the Makefile, but the operating system wasn't taking it
	#This is a bit forced, but it works. 
	#Each time, I just modify the Makefile to pass appropriate arguments into the C program
	#regular gcc wasn't working
	with open("Makefile_bkup","r") as Make:
		lines = Make.readlines()
		lines[2] = lines[2].strip('\n')
		lines[2] = lines[2] + " -DIMAGE_W=" + str(dimension) + " -DIMAGE_H=" + str(dimension) + " -DFILTER_W=" + str(filter_size)
		Make.close()
	with open("Makefile","w") as Make:
		Make.writelines(lines)
		Make.close()

	compileCmd = "make "

    #print compileCmd

    # Compile the code
	os.system(compileCmd)

    # Run the code and find timing data
	runCmd =  "./convolveCL"
	os.system(runCmd)

    # The following captures the output unlike the os.system control
	proc = subprocess.Popen(runCmd, shell=True, bufsize=256, stdout=subprocess.PIPE)
	for line in proc.stdout:
		line = line.strip('\n')
		test = line.split(":")
		timing_data.append([dimension,filter_size,test[0],int(test[1])] )

for dimension in [512,1024,2048]:
	print "\n"+str(dimension)
	for filter_size in [3,5,7,9,11,13]:
		runtest(dimension,filter_size)

with open("data.txt","w") as file:
	for item in timing_data:
		toFile = str(item)
		toFile = toFile.strip(']')
		toFile = toFile.strip('[')
		file.write(toFile)
		file.write('\n')
	file.close()

