
from pathlib import Path
import shutil

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    LongType
)
from pyspark.sql.functions import col, round as spark_round


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CREATE SPARK SESSION
# ==========================================

spark = (
    SparkSession.builder
    .appName("TourismIntelligence-DataIngestion")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\n==========================================")
print("   TOURISM INTELLIGENCE SYSTEM")
print("   PYSPARK DATA INGESTION")
print("==========================================")

try:

    # ==========================================
    # 3. DEFINE DATA SCHEMAS
    # ==========================================

    annual_schema = StructType([
        StructField("year", IntegerType(), False),
        StructField("state", StringType(), False),
        StructField("domestic_tourist_visits", LongType(), True),
        StructField("foreign_tourist_visits", LongType(), True)
    ])

    monthly_schema = StructType([
        StructField("year", IntegerType(), False),
        StructField("month", IntegerType(), False),
        StructField("state", StringType(), False),
        StructField("tourist_arrivals", LongType(), True)
    ])

    hotel_schema = StructType([
        StructField("date", StringType(), False),
        StructField("state", StringType(), False),
        StructField("destination", StringType(), False),
        StructField("total_rooms", IntegerType(), True),
        StructField("occupied_rooms", IntegerType(), True)
    ])


    # ==========================================
    # 4. LOAD CSV DATASETS
    # ==========================================

    print("\nLoading tourism datasets...")

    annual_df = (
        spark.read
        .option("header", True)
        .schema(annual_schema)
        .csv(str(RAW_DIR / "tourist_arrivals.csv"))
    )

    monthly_df = (
        spark.read
        .option("header", True)
        .schema(monthly_schema)
        .csv(str(RAW_DIR / "monthly_tourist_arrivals.csv"))
    )

    hotel_df = (
        spark.read
        .option("header", True)
        .schema(hotel_schema)
        .csv(str(RAW_DIR / "hotel_occupancy.csv"))
    )

    print("All datasets loaded successfully!")


    # ==========================================
    # 5. DATA CLEANING
    # ==========================================

    print("\nCleaning datasets...")

    # Annual Tourist Arrivals
    annual_df = (
        annual_df
        .dropna(subset=["year", "state"])
        .filter(
            (col("domestic_tourist_visits") >= 0) &
            (col("foreign_tourist_visits") >= 0)
        )
        .dropDuplicates()
    )

    # Monthly Tourist Arrivals
    monthly_df = (
        monthly_df
        .dropna(
            subset=[
                "year",
                "month",
                "state",
                "tourist_arrivals"
            ]
        )
        .filter(
            (col("month") >= 1) &
            (col("month") <= 12) &
            (col("tourist_arrivals") >= 0)
        )
        .dropDuplicates()
    )

    # Hotel Occupancy
    hotel_df = (
        hotel_df
        .dropna(
            subset=[
                "date",
                "state",
                "destination",
                "total_rooms",
                "occupied_rooms"
            ]
        )
        .filter(
            (col("total_rooms") > 0) &
            (col("occupied_rooms") >= 0) &
            (col("occupied_rooms") <= col("total_rooms"))
        )
        .withColumn(
            "occupancy_rate_pct",
            spark_round(
                col("occupied_rooms") /
                col("total_rooms") * 100,
                2
            )
        )
        .dropDuplicates()
    )

    print("Data cleaning completed!")


    # ==========================================
    # 6. DISPLAY DATASETS
    # ==========================================

    print("\n========== ANNUAL TOURIST ARRIVALS ==========")

    annual_df.show(truncate=False)
    annual_df.printSchema()

    print("\n========== MONTHLY TOURIST ARRIVALS ==========")

    monthly_df.show(truncate=False)
    monthly_df.printSchema()

    print("\n========== HOTEL OCCUPANCY ==========")

    hotel_df.show(truncate=False)
    hotel_df.printSchema()


    # ==========================================
    # 7. DISPLAY RECORD COUNTS
    # ==========================================

    print("\n========== DATASET SUMMARY ==========")

    print("Annual Records:", annual_df.count())
    print("Monthly Records:", monthly_df.count())
    print("Hotel Records:", hotel_df.count())


    # ==========================================
    # 8. EXPORT CLEANED DATASETS
    # ==========================================

    print("\nExporting cleaned datasets...")

    # Convert Spark DataFrames to Pandas
    annual_pd = annual_df.toPandas()
    monthly_pd = monthly_df.toPandas()
    hotel_pd = hotel_df.toPandas()

    # Output file paths
    output_files = [
        PROCESSED_DIR / "tourist_arrivals_clean.csv",
        PROCESSED_DIR / "monthly_tourist_arrivals_clean.csv",
        PROCESSED_DIR / "hotel_occupancy_clean.csv"
    ]

    # Remove previous Spark output directories/files if present
    for path in output_files:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    # Save as normal CSV files
    annual_pd.to_csv(
        output_files[0],
        index=False
    )

    monthly_pd.to_csv(
        output_files[1],
        index=False
    )

    hotel_pd.to_csv(
        output_files[2],
        index=False
    )

    print("\nAll CSV files exported successfully!")


    # ==========================================
    # 9. FINAL SUMMARY
    # ==========================================

    print("\n==========================================")
    print("       DATA INGESTION COMPLETED")
    print("==========================================")

    print("Annual Dataset:", output_files[0])
    print("Monthly Dataset:", output_files[1])
    print("Hotel Dataset:", output_files[2])

    print("\nTotal Records Processed:",
          len(annual_pd) +
          len(monthly_pd) +
          len(hotel_pd))

    print("\nProject executed successfully!")


finally:

    # ==========================================
    # 10. STOP SPARK SESSION
    # ==========================================

    spark.stop()

    print("\nSpark Session Closed Successfully.")
