from __future__ import print_function
from pyspark import SparkContext, SparkConf
from operator import add
from pyspark.mllib.feature import HashingTF, IDF # TF-IDF specific functions 
import sys
import os
import operator
from time import time

sc = SparkContext("local", "TF-IDF: Data analysis with Spark")
sc.setLogLevel("ERROR")


# Import the various csv files used
#   Listings_us.csv
fsListings = sc.textFile('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/listings_us.csv', use_unicode = True)
listingHeader = fsListings.first()
listingsFiltered = fsListings.filter(lambda x: x!=listingHeader).map(lambda x: x.split("\t"))

#   calendar_us.csv
fsCalendar = sc.textFile('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/calendar_us.csv', use_unicode = False)
calendarHeader = fsCalendar.first()
calendarFiltered = fsCalendar.filter(lambda x: x!=calendarHeader).map(lambda x: x.split("\t"))





# Checking flags, 
#   listing (-l) or a neighborhood (-n) 
#   should be analysed and listing id or neighborhood name on the input.
def flagPassing(args):
    for arg in range(len(args)):
        if args[arg]=='-l':
            listingID = args[arg+1]
            print('Flag -l accepted. Checking listingID: '+args[arg+1])
            start_time = time()
            idf2(tf(heyListen(listingID)), 1)
            print("Checking listing Elapsed time: " + str(time() - start_time))
            #tfIDF(listingAndDescription[listingID])
        elif args[arg] == '-n':
            neighbor = args[arg+1]
            print('Flag -n accepted. Checking neighborhood: '+neighbor)
            start_time = time()
            idf2(tf(heyNeighbor(neighbor)), 2)
            print("Checking neighbourhood Elapsed time: " + str(time() - start_time))

