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





# Checking running parameters, 
#   alternative_listings <listing_id> <date:YYY-MM-DD> <x> <y> <n>
#       where   
#               x = price is not higher than x%
#               y = radius y km
#               n = output top n listings with common amenities as <listing_id>
#   Example:
#   alternative_listings.py 15359479 2016-12-15 10 2 20
def parametersPassing(args):
    listing_id = args[1]
    date = args[2]    
    x = args[3]
    y = args[4]
    n = args[5]
    

