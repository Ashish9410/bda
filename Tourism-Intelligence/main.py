
from pyspark.sql import SparkSession

# Create Spark Session
spark = (
    SparkSession.builder
    .appName("Tourism Intelligence System")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

# Display project information
print("======================================")
print(" TOURISM INTELLIGENCE SYSTEM")
print("======================================")

print("Spark Version:", spark.version)
print("Application Name:", spark.sparkContext.appName)

# Create sample tourism data
data = [
    ("Goa", 85000),
    ("Manali", 65000),
    ("Jaipur", 72000),
    ("Kerala", 90000),
    ("Agra", 78000)
]

columns = ["Destination", "Tourist_Arrivals"]

df = spark.createDataFrame(data, columns)

print("\nTourism Dataset:")
df.show()

# Calculate total tourists
print("Total Tourist Arrivals:")

df.selectExpr(
    "SUM(Tourist_Arrivals) AS Total_Tourists"
).show()

# Stop Spark
spark.stop()

print("Project executed successfully!")
