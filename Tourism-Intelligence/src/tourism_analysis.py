
from pathlib import Path

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CREATE SPARK SESSION
# ==========================================

spark = (
    SparkSession.builder
    .appName("TourismIntelligence-EDA")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\n======================================")
print(" TOURISM INTELLIGENCE - DATA ANALYSIS")
print("======================================")


try:

    # ==========================================
    # 3. LOAD CLEANED DATASETS
    # ==========================================

    annual_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_DIR / "tourist_arrivals_clean.csv"))
    )

    monthly_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_DIR / "monthly_tourist_arrivals_clean.csv"))
    )

    hotel_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_DIR / "hotel_occupancy_clean.csv"))
    )

    print("\nAll datasets loaded successfully!")

    # ==========================================
    # 4. STATE-WISE TOURIST ARRIVALS
    # ==========================================

    print("\n1. STATE-WISE TOURIST ARRIVALS")

    state_analysis = (
        annual_df
        .groupBy("state")
        .agg(
            F.sum("domestic_tourist_visits")
            .alias("domestic_visitors"),

            F.sum("foreign_tourist_visits")
            .alias("foreign_visitors")
        )
        .withColumn(
            "total_visitors",
            F.col("domestic_visitors") +
            F.col("foreign_visitors")
        )
        .orderBy(
            F.desc("total_visitors")
        )
    )

    state_analysis.show(truncate=False)

    # ==========================================
    # 5. DESTINATION POPULARITY
    # ==========================================

    print("\n2. TOP TOURIST DESTINATIONS")

    destination_analysis = (
        state_analysis
        .orderBy(F.desc("total_visitors"))
    )

    destination_analysis.show(truncate=False)

    # ==========================================
    # 6. DOMESTIC VS FOREIGN TOURISM
    # ==========================================

    print("\n3. DOMESTIC VS FOREIGN TOURISM")

    tourism_comparison = (
        annual_df
        .agg(
            F.sum("domestic_tourist_visits")
            .alias("total_domestic"),

            F.sum("foreign_tourist_visits")
            .alias("total_foreign")
        )
        .withColumn(
            "total_visitors",
            F.col("total_domestic") +
            F.col("total_foreign")
        )
        .withColumn(
            "domestic_percentage",
            F.round(
                F.col("total_domestic") /
                F.col("total_visitors") * 100,
                2
            )
        )
        .withColumn(
            "foreign_percentage",
            F.round(
                F.col("total_foreign") /
                F.col("total_visitors") * 100,
                2
            )
        )
    )

    tourism_comparison.show(truncate=False)

    # ==========================================
    # 7. MONTHLY TOURISM TRENDS
    # ==========================================

    print("\n4. MONTHLY TOURISM TRENDS")

    monthly_analysis = (
        monthly_df
        .groupBy("year", "month")
        .agg(
            F.sum("tourist_arrivals")
            .alias("total_arrivals")
        )
        .orderBy("year", "month")
    )

    monthly_analysis.show(truncate=False)

    # State-wise monthly trends
    state_monthly_analysis = (
        monthly_df
        .groupBy("year", "month", "state")
        .agg(
            F.sum("tourist_arrivals")
            .alias("total_arrivals")
        )
        .orderBy("year", "month", "state")
    )

    # ==========================================
    # 8. HOTEL OCCUPANCY ANALYSIS
    # ==========================================

    print("\n5. HOTEL OCCUPANCY ANALYSIS")

    hotel_analysis = (
        hotel_df
        .groupBy("state", "destination")
        .agg(
            F.sum("total_rooms").alias("total_rooms"),

            F.sum("occupied_rooms").alias("occupied_rooms")
        )
        .withColumn(
            "occupancy_rate_pct",
            F.round(
                F.col("occupied_rooms") /
                F.col("total_rooms") * 100,
                2
            )
        )
        .orderBy(
            F.desc("occupancy_rate_pct")
        )
    )

    hotel_analysis.show(truncate=False)

    # ==========================================
    # 9. YEAR-OVER-YEAR TOURISM GROWTH
    # ==========================================

    print("\n6. TOURISM GROWTH ANALYSIS")

    annual_totals = (
        annual_df
        .withColumn(
            "total_visitors",
            F.col("domestic_tourist_visits") +
            F.col("foreign_tourist_visits")
        )
        .groupBy("state", "year")
        .agg(
            F.sum("total_visitors")
            .alias("total_visitors")
        )
    )

    growth_window = (
        Window
        .partitionBy("state")
        .orderBy("year")
    )

    growth_analysis = (
        annual_totals
        .withColumn(
            "previous_year_visitors",
            F.lag("total_visitors", 1)
            .over(growth_window)
        )
        .withColumn(
            "growth_percentage",
            F.when(
                F.col("previous_year_visitors").isNotNull()
                & (F.col("previous_year_visitors") != 0),
                F.round(
                    (
                        F.col("total_visitors") -
                        F.col("previous_year_visitors")
                    ) /
                    F.col("previous_year_visitors") * 100,
                    2
                )
            )
        )
        .orderBy("state", "year")
    )

    growth_analysis.show(truncate=False)

    # ==========================================
    # 10. EXPORT RESULTS FOR POWER BI
    # ==========================================

    print("\n7. EXPORTING ANALYSIS RESULTS")

    analysis_outputs = {
        "state_wise_analysis.csv": state_analysis,
        "destination_popularity.csv": destination_analysis,
        "tourism_comparison.csv": tourism_comparison,
        "monthly_trends.csv": monthly_analysis,
        "state_monthly_trends.csv": state_monthly_analysis,
        "hotel_analysis.csv": hotel_analysis,
        "tourism_growth.csv": growth_analysis
    }

    for filename, dataframe in analysis_outputs.items():

        output_path = OUTPUT_DIR / filename

        dataframe.toPandas().to_csv(
            output_path,
            index=False
        )

        print("Exported:", filename)

    print("\n======================================")
    print(" EXPLORATORY DATA ANALYSIS COMPLETED")
    print("======================================")

    print("All analysis files saved in:")
    print(OUTPUT_DIR)

finally:

    spark.stop()

    print("\nSpark Session Closed Successfully.")
