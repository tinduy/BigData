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
from math import radians, sin, cos, sqrt, asin

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

# Maps index name to column index for listings_us.csv
# getIndexValue then uses the key (column name) to access right column index
header2 = fsListings.first().split("\t")
dict = {}
for i in range(len(header2)):
    dict[header2[i]] = i

# getIndexValue then uses the key (column name) to access right column index
def getIndexValue(name):
    return dict[name]

listingColumns = listingsFiltered.map(lambda x: (x[getIndexValue("id")], x[getIndexValue("room_type")],\
                            float(x[getIndexValue("price")].replace("$", "").replace(",", "")),\
                            float(x[getIndexValue("longitude")]), float(x[getIndexValue("latitude")]),\
                                  x[getIndexValue("amenities")], x[getIndexValue("name")]))

def alternativeListings(listingID, date, pricePercentage, y, n):
    start_time = time()
    # Filter out the available listings on date
    filterAvailable = calendarFiltered.filter(lambda line: date == line[1]).filter(lambda line: line[2] == 't').map(lambda x: (x[0], x[2]))
    roomType = listingColumns.map(lambda x: (x[0], x[1]))
    # Join with key = id
    joinRDD = filterAvailable.join(roomType)
    listingRoomType = getRoomType(listingID)
    # Filter out the listings with same room type as listingID
    roomTypeRDD = joinRDD.filter(lambda line: line[1][1] == listingRoomType)
    # Join with key = id
    priceRDD = roomTypeRDD.map(lambda x: (x[0], x[1][1])).join(listingColumns.map(lambda x: (x[0], x[2])))
    # Calculate the maximum price for the listings
    maxPriceForListing = float(float(listingColumns.filter(lambda line: listingID == line[0]).\
                                     map(lambda x: x[2]).collect()[0]) * float(pricePercentage))
    # Filter out the remaining listings with price less than the maximum price
    checkPrice = priceRDD.filter(lambda x: x[1][1] < maxPriceForListing)
    longLat = checkPrice.map(lambda x: (x[0], x[1][1])).\
                        join(listingColumns.map(lambda x: (x[0], x[3]))).\
                        map(lambda x: (x[0], x[1][1])).\
                        join(listingColumns.map(lambda x: (x[0], x[4])))
    # Get the coordinates from the original listing (listingID)
    listingCoordinates = listingColumns.map(lambda x: (x[0], float(x[3]), float(x[4]))).filter(lambda id: listingID in id)
    long = listingCoordinates.map(lambda x: x[1]).collect()[0]
    lat = listingCoordinates.map(lambda x: x[2]).collect()[0]
    # Calculate the distances between coordinates by using the function haversine()
    distanceCalculation = longLat.map(lambda x: (x[0], x[1][0], x[1][1])).\
                                map(lambda x: (x[0], haversine(float(x[2]), float(x[1]), float(lat), float(long))))
    # Filter out the listings that are less than y km away from listingID
    filterDistance = distanceCalculation.filter(lambda x: x[1] < float(y))
    # Get the amenities for original listingID
    amenitiesListing = listingColumns.map(lambda x: (x[0], x[5])).\
                                    filter(lambda id: listingID == id[0]).\
                                    map(lambda x: x[1].replace("{", "").replace("}", "").\
                                    replace("\"", "").lower().strip().split(",")).collect()[0]
    # Get the amenities for the remaining listings, join with key = id
    listAmenities = filterDistance.join(listingColumns.map(lambda x: (x[0], x[5]))).\
                                                    map(lambda x: (x[0], x[1][1].replace("{", "").replace("}", "").\
                                                    replace("\"", "").lower().strip().split(",")))
    # Check amenities up against the amenities list from listingID
    checkAmenities = listAmenities.flatMapValues(lambda x: x).\
                                    filter(lambda x: x[1] in amenitiesListing).\
                                    map(lambda x: (x[0], int(1))).\
                                    reduceByKey(add).\
                                    map(lambda x: (x[1], x[0])).\
                                    sortByKey(0,1).\
                                    map(lambda x: (x[1], x[0])).\
                                    take(int(n))
    #print(checkAmenities)
    listId = sc.parallelize(checkAmenities).map(lambda x: x[0]).collect()
    amenities = sc.parallelize(checkAmenities).collectAsMap()
    distance = filterDistance.collectAsMap()
    # Output listings into file called alternatives.tsv
    fileMaker = listingColumns.filter(lambda id: id[0] in listId).map(lambda x: (x[0], x[6],\
                                                        amenities[x[0]], distance[x[0]], x[2])).\
                                                        map(lambda x: (x[2], x[0], x[1], x[3], x[4])).\
                                                        sortByKey(0,1).\
                                                        map(lambda x: (x[1], x[2], x[0], x[3], x[4]))
    fileMaker.map(lambda x: "\t".join(map(str, x))).coalesce(1).saveAsTextFile("/usr/local/spark/spark-2.1.0-bin-hadoop2.7/alternatives2.tsv")
    print("Checking listing Elapsed time: " + str(time() - start_time))
    #Make file for CARTO vizualisation
    cartoMaker = listingColumns.filter(lambda id: id[0] in listId).map(lambda x: (x[0], x[6],\
                                                        amenities[x[0]], distance[x[0]], x[2], x[3], x[4])).\
                                                        map(lambda x: (x[2], x[0], x[1], x[3], x[4], x[5], x[6])).\
                                                        sortByKey(0, 1).\
                                                        map(lambda x: (x[1], x[2], x[0], x[3], x[4], x[5], x[6]))
    # Original listingID columns needed
    listingMaker = listingColumns.filter(lambda id: listingID in id).map(lambda x: (x[0], x[6], 0, 0, x[2], x[3], x[4]))
    #cartoMaker.map(lambda x: "\t".join(map(str, x))).coalesce(1).saveAsTextFile("/usr/local/spark/spark-2.1.0-bin-hadoop2.7/cartoAlternatives.tsv")

# Get room type for listingID
def getRoomType(listingID):
    room_type = listingColumns.map(lambda x: (x[0],x[1])).filter(lambda id: listingID in id).map(lambda x: x[1]).collect()
    return room_type[0]

# Calculate distances between coordinates
def haversine(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


# Checking running parameters, 
#   alternative_listings <listingID> <date:YYY-MM-DD> <x> <y> <n>
#       where   
#               x = price is not higher than x%
#               y = radius y km
#               n = output top n listings with common amenities as <listingID>
#   Example:
#   alternative_listings.py 4717459 2017-05-28 10 1 50
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
    (alternativeListings(listingID,date,x,y,n))

'''    ------------------ When running, under here  ---------------------	 '''

# Skeleton code for standalone application

print("2 Finding Alternative listings")
parametersPassing(sys.argv)








