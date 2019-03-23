package executor

import scala.collection.mutable.HashMap

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.monotonically_increasing_id

import appraisal.spark.eraser.Eraser

object KnnExec {

  def main(args: Array[String]) {

    try {

      Logger.getLogger("org").setLevel(Level.ERROR)

      val spark = SparkSession
        .builder
        .appName("LinearRegressionDF")
        .master("local[*]")
        .config("spark.sql.warehouse.dir", "file:///C:/temp") // Necessary to work around a Windows bug in Spark 2.0.0; omit if you're not on Windows.
        .getOrCreate()

      var df = utility.Utility.loadDataset(spark, "iris.data.csv")

      val percent = (10, 20, 30, 40, 50)

      val attributes = Array[String](
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class")

      val idf = Eraser.run(df, attributes(4), percent._1).withColumn("lineId", monotonically_increasing_id)

      val params: HashMap[String, Any] = HashMap("attributes" -> attributes, "k" -> 10)

      val imputationResult = algorithm.Knn.run(idf, percent._1, attributes(4), params)

      val sImputationResult = statistic.Statistic.statisticInfo(df, attributes(4), imputationResult)

      Logger.getLogger("appraisal").info("Algoritmo: k-NN")

      sImputationResult.result.sortBy(_.lineId).collect().foreach(x => Logger.getLogger("appraisal").info(s"originalValue: ${x.originalValue} -> imputationValue: ${x.imputationValue} -> Error: ${x.error}"))

      Logger.getLogger("appraisal").info("totalError: " + sImputationResult.totalError)
      Logger.getLogger("appraisal").info("avgPercentError: " + sImputationResult.avgPercentError)

    } catch {

      case ex: Exception => Logger.getLogger("appraisal").error(ex)

    }

  }

}