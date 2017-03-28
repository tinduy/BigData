from __future__ import print_function
from pyspark import SparkContext, SparkConf
from operator import add
from pyspark.mllib.feature import HashingTF, IDF # TF-IDF specific functions 
import sys
import os
import operator
from time import time
import csv
from pyspark.sql import SQLContext

sc = SparkContext("local[4]", "TF-IDF: Data analysis with Spark")
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

# Import the various csv files used
#   Listings_us.csv
fsListings = sc.textFile('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/listings_us.csv', use_unicode = False)
listingHeader = fsListings.first()
listingsFiltered = fsListings.filter(lambda x: x!=listingHeader).map(lambda x: x.split("\t"))

#   calendar_us.csv
fsCalendar = sc.textFile('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/calendar_us.csv', use_unicode = False)
calendarHeader = fsCalendar.first()
calendarFiltered = fsCalendar.filter(lambda x: x!=calendarHeader).map(lambda x: x.split("\t"))

header2 = fsListings.first().split("\t")
dict = {}
for i in range(len(header2)):
    dict[header2[i]] = i

# getIndexValue then uses the key (column name) to access right column index.
def getIndexValue(name):
    return dict[name]

listingColumns = listingsFiltered.map(lambda x: (x[getIndexValue("id")], x[getIndexValue("room_type")], \
                            float(x[getIndexValue("price")].replace("$", "").replace(",", "")), \
                            float(x[getIndexValue("longitude")]), float(x[getIndexValue("latitude")])))
#print(listingColumns.take(10))

def checkAvailable(listingID, date, pricePercentage):
    start_time = time()
    filterAvailable = calendarFiltered.filter(lambda line: date == line[1]).filter(lambda line: line[2] == 't').map(lambda x: (x[0], x[2]))
    roomType = listingColumns.map(lambda x: (x[0], x[1]))
    joinRDD = filterAvailable.join(roomType)
    listingRoomType = getRoomType(listingID)
    roomTypeRDD = joinRDD.filter(lambda line: line[1][1] == listingRoomType)
    priceRDD = roomTypeRDD.map(lambda x: (x[0], x[1][1])).join(listingColumns.map(lambda x: (x[0], x[2])))
    maxPriceForListing = float(float(listingColumns.filter(lambda line: listingID == line[0]).map(lambda x: x[2]).collect()[0]) * float(pricePercentage))
    checkPrice = priceRDD.filter(lambda x: x[1][1] < maxPriceForListing)
    longLat = checkPrice.map(lambda x: (x[0], x[1][1])).\
                        join(listingColumns.map(lambda x: (x[0], x[3]))).\
                        map(lambda x: (x[0], x[1][1])).\
                        join(listingColumns.map(lambda x: (x[0], x[4])))
    #longLat.map(lambda x: (x[0], x[1][0], x[1][1])).toDF().coalesce(1).write.csv('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/longLat.csv')
    longLatFiltered = sc.textFile('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/longLat.csv')

    print("Checking listing Elapsed time: " + str(time() - start_time))



def findAlternativeListing(listingID, room_type):
    return None


def getRoomType(listingID):
    room_type = listingColumns.map(lambda x: (x[0],x[1])).filter(lambda id: listingID in id).map(lambda x: x[1]).collect()
    return room_type[0]





# Checking running parameters, 
#   alternative_listings <listingID> <date:YYY-MM-DD> <x> <y> <n>
#       where   
#               x = price is not higher than x%
#               y = radius y km
#               n = output top n listings with common amenities as <listingID>
#   Example:
#   alternative_listings.py 15359479 2016-12-15 10 2 20
def parametersPassing(args):
    listingID = args[1]
    date = args[2]    
    x = (float(args[3])/100)+1
    y = args[4]
    n = args[5]
    print("Checking listing id \t"+listingID)
    print("On date \t\t"+date)
    print("\nIf alternative listing:")
    print("Alt. listing not exceeding price of\t"+str(x)+"")
    print("Within a radius of \t\t\t"+y+'KM')
    print("Displaying top n=\t\t\t"+n+" listings")
    
    if (checkAvailable(listingID,date,x)):
        print("A okay, not occupied here, book it before it's too late")
    else:
        room_type=getRoomType(listingID)#getRoomType('4717459')
        #findAlternativeListing
        print("room is not available")
    

'''    ------------------ When running, under here  ---------------------	 '''

# Skeleton code for standalone application

print("2 Finding Alternative listings")
parametersPassing(sys.argv)








