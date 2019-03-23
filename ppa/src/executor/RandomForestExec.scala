package executor

import scala.collection.mutable.HashMap

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.functions.udf

import appraisal.spark.eraser.Eraser

object RandomForestExec {

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

      val params: HashMap[String, Any] = HashMap("attributes" -> attributes, "numClasses" -> 4, "numTrees" -> 10, "featureSubsetStrategy" -> "auto", "impurity" -> "gini", "maxDepth" -> 5, "maxBins" -> 32)

      def convertIrisToDouble(s: String): Double = {
        val iris = s match {
          case "Iris-setosa"     => 1.0
          case "Iris-versicolor" => 2.0
          case "Iris-virginica"  => 3.0
          case _                 => 0.0
        }

        iris
      }

      val irisCallback = (i: String) => { convertIrisToDouble(i) }

      val imputationResult = algorithm.RandomForest.run(idf, attributes(4), params, irisCallback)

      val convertIrisToDoubleUdf = udf[Double, String](convertIrisToDouble)

      val mdf = df
        .withColumn(attributes(0), df(attributes(0)))
        .withColumn(attributes(1), df(attributes(1)))
        .withColumn(attributes(2), df(attributes(2)))
        .withColumn(attributes(3), df(attributes(3)))
        .withColumn(attributes(4), convertIrisToDoubleUdf(df(attributes(4))))

      val sImputationResult = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), imputationResult)

      Logger.getLogger("appraisal").info("Algoritmo: Random forest")

      sImputationResult.result.sortBy(_.lineId).collect().foreach(x => Logger.getLogger("appraisal").info(s"originalValue: ${x.originalValue} -> imputationValue: ${x.imputationValue} -> Error: ${x.error}"))

      Logger.getLogger("appraisal").info("totalError: " + sImputationResult.totalError)
      Logger.getLogger("appraisal").info("avgPercentError: " + sImputationResult.avgPercentError)

    } catch {

      case ex: Exception => Logger.getLogger("appraisal").error(ex)

    }

  }

}