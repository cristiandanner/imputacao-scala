package executor

import scala.collection.mutable.HashMap

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.functions.udf

import appraisal.spark.eraser.Eraser

object CompareIrisExec {

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

      val idf10 = Eraser.run(df, attributes(4), percent._1).withColumn("lineId", monotonically_increasing_id) // 10% absence
      val idf20 = Eraser.run(df, attributes(4), percent._2).withColumn("lineId", monotonically_increasing_id) // 20% absence
      val idf30 = Eraser.run(df, attributes(4), percent._3).withColumn("lineId", monotonically_increasing_id) // 30% absence
      val idf40 = Eraser.run(df, attributes(4), percent._4).withColumn("lineId", monotonically_increasing_id) // 40% absence
      val idf50 = Eraser.run(df, attributes(4), percent._5).withColumn("lineId", monotonically_increasing_id) // 50% absence

      val params: HashMap[String, Any] = HashMap("attributes" -> attributes, "k" -> 10, "numClasses" -> 4, "numTrees" -> 10, "blockSize" -> 128, "seed" -> 1234L, "maxIter" -> 145, "featureSubsetStrategy" -> "auto", "impurity" -> "gini", "maxDepth" -> 5, "maxBins" -> 32)

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

      val knnImputationResult10 = algorithm.Knn.run(idf10, percent._1, attributes(4), params)
      val knnImputationResult20 = algorithm.Knn.run(idf20, percent._2, attributes(4), params)
      val knnImputationResult30 = algorithm.Knn.run(idf30, percent._3, attributes(4), params)
      val knnImputationResult40 = algorithm.Knn.run(idf40, percent._4, attributes(4), params)
      val knnImputationResult50 = algorithm.Knn.run(idf50, percent._5, attributes(4), params)

      val decisionTreeImputationResult10 = algorithm.DecisionTree.run(idf10, attributes(4), params, irisCallback)
      val decisionTreeImputationResult20 = algorithm.DecisionTree.run(idf20, attributes(4), params, irisCallback)
      val decisionTreeImputationResult30 = algorithm.DecisionTree.run(idf30, attributes(4), params, irisCallback)
      val decisionTreeImputationResult40 = algorithm.DecisionTree.run(idf40, attributes(4), params, irisCallback)
      val decisionTreeImputationResult50 = algorithm.DecisionTree.run(idf50, attributes(4), params, irisCallback)

      val randomForestImputationResult10 = algorithm.RandomForest.run(idf10, attributes(4), params, irisCallback)
      val randomForestImputationResult20 = algorithm.RandomForest.run(idf20, attributes(4), params, irisCallback)
      val randomForestImputationResult30 = algorithm.RandomForest.run(idf30, attributes(4), params, irisCallback)
      val randomForestImputationResult40 = algorithm.RandomForest.run(idf40, attributes(4), params, irisCallback)
      val randomForestImputationResult50 = algorithm.RandomForest.run(idf50, attributes(4), params, irisCallback)

      val multilayerImputationResult10 = algorithm.MultilayerPerceptron.run(idf10, attributes(4), params, irisCallback)
      val multilayerImputationResult20 = algorithm.MultilayerPerceptron.run(idf20, attributes(4), params, irisCallback)
      val multilayerImputationResult30 = algorithm.MultilayerPerceptron.run(idf30, attributes(4), params, irisCallback)
      val multilayerImputationResult40 = algorithm.MultilayerPerceptron.run(idf40, attributes(4), params, irisCallback)
      val multilayerImputationResult50 = algorithm.MultilayerPerceptron.run(idf50, attributes(4), params, irisCallback)

      val convertIrisToDoubleUdf = udf[Double, String](convertIrisToDouble)

      val mdf = df
        .withColumn(attributes(0), df(attributes(0)))
        .withColumn(attributes(1), df(attributes(1)))
        .withColumn(attributes(2), df(attributes(2)))
        .withColumn(attributes(3), df(attributes(3)))
        .withColumn(attributes(4), convertIrisToDoubleUdf(df(attributes(4))))

      val knnSImputationResult10 = statistic.Statistic.statisticInfo(df, attributes(4), knnImputationResult10)
      val knnSImputationResult20 = statistic.Statistic.statisticInfo(df, attributes(4), knnImputationResult20)
      val knnSImputationResult30 = statistic.Statistic.statisticInfo(df, attributes(4), knnImputationResult30)
      val knnSImputationResult40 = statistic.Statistic.statisticInfo(df, attributes(4), knnImputationResult40)
      val knnSImputationResult50 = statistic.Statistic.statisticInfo(df, attributes(4), knnImputationResult50)

      val decisionTreeSImputationResult10 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), decisionTreeImputationResult10)
      val decisionTreeSImputationResult20 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), decisionTreeImputationResult20)
      val decisionTreeSImputationResult30 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), decisionTreeImputationResult30)
      val decisionTreeSImputationResult40 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), decisionTreeImputationResult40)
      val decisionTreeSImputationResult50 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), decisionTreeImputationResult50)

      val randomForestSImputationResult10 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), randomForestImputationResult10)
      val randomForestSImputationResult20 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), randomForestImputationResult20)
      val randomForestSImputationResult30 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), randomForestImputationResult30)
      val randomForestSImputationResult40 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), randomForestImputationResult40)
      val randomForestSImputationResult50 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), randomForestImputationResult50)

      val multilayerSImputationResult10 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), multilayerImputationResult10)
      val multilayerSImputationResult20 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), multilayerImputationResult20)
      val multilayerSImputationResult30 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), multilayerImputationResult30)
      val multilayerSImputationResult40 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), multilayerImputationResult40)
      val multilayerSImputationResult50 = statistic.Statistic.numericStatisticInfo(mdf, attributes(4), multilayerImputationResult50)

      Logger.getLogger("appraisal").info("===========================================================")

      Logger.getLogger("appraisal").info("Algoritmo: k-NN - 10% absence")
      Logger.getLogger("appraisal").info("k-NN - totalError: " + knnSImputationResult10.totalError)
      Logger.getLogger("appraisal").info("k-NN - avgPercentError: " + knnSImputationResult10.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: k-NN - 20% absence")
      Logger.getLogger("appraisal").info("k-NN - totalError: " + knnSImputationResult20.totalError)
      Logger.getLogger("appraisal").info("k-NN - avgPercentError: " + knnSImputationResult20.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: k-NN - 30% absence")
      Logger.getLogger("appraisal").info("k-NN - totalError: " + knnSImputationResult30.totalError)
      Logger.getLogger("appraisal").info("k-NN - avgPercentError: " + knnSImputationResult30.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: k-NN - 40% absence")
      Logger.getLogger("appraisal").info("k-NN - totalError: " + knnSImputationResult40.totalError)
      Logger.getLogger("appraisal").info("k-NN - avgPercentError: " + knnSImputationResult40.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: k-NN - 50% absence")
      Logger.getLogger("appraisal").info("k-NN - totalError: " + knnSImputationResult50.totalError)
      Logger.getLogger("appraisal").info("k-NN - avgPercentError: " + knnSImputationResult50.avgPercentError)

      Logger.getLogger("appraisal").info("===========================================================")

      Logger.getLogger("appraisal").info("Algoritmo: Decision tree - 10% absence")
      Logger.getLogger("appraisal").info("Decision tree - totalError: " + decisionTreeSImputationResult10.totalError)
      Logger.getLogger("appraisal").info("Decision tree - avgPercentError: " + decisionTreeSImputationResult10.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: Decision tree - 20% absence")
      Logger.getLogger("appraisal").info("Decision tree - totalError: " + decisionTreeSImputationResult20.totalError)
      Logger.getLogger("appraisal").info("Decision tree - avgPercentError: " + decisionTreeSImputationResult20.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: Decision tree - 30% absence")
      Logger.getLogger("appraisal").info("Decision tree - totalError: " + decisionTreeSImputationResult30.totalError)
      Logger.getLogger("appraisal").info("Decision tree - avgPercentError: " + decisionTreeSImputationResult30.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: Decision tree - 40% absence")
      Logger.getLogger("appraisal").info("Decision tree - totalError: " + decisionTreeSImputationResult40.totalError)
      Logger.getLogger("appraisal").info("Decision tree - avgPercentError: " + decisionTreeSImputationResult40.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: Decision tree - 50% absence")
      Logger.getLogger("appraisal").info("Decision tree - totalError: " + decisionTreeSImputationResult50.totalError)
      Logger.getLogger("appraisal").info("Decision tree - avgPercentError: " + decisionTreeSImputationResult50.avgPercentError)

      Logger.getLogger("appraisal").info("===========================================================")

      Logger.getLogger("appraisal").info("Algoritmo: Random forest - 10% absence")
      Logger.getLogger("appraisal").info("Random forest - totalError: " + randomForestSImputationResult10.totalError)
      Logger.getLogger("appraisal").info("Random forest - avgPercentError: " + randomForestSImputationResult10.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: Random forest - 20% absence")
      Logger.getLogger("appraisal").info("Random forest - totalError: " + randomForestSImputationResult20.totalError)
      Logger.getLogger("appraisal").info("Random forest - avgPercentError: " + randomForestSImputationResult20.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: Random forest - 30% absence")
      Logger.getLogger("appraisal").info("Random forest - totalError: " + randomForestSImputationResult30.totalError)
      Logger.getLogger("appraisal").info("Random forest - avgPercentError: " + randomForestSImputationResult30.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: Random forest - 40% absence")
      Logger.getLogger("appraisal").info("Random forest - totalError: " + randomForestSImputationResult40.totalError)
      Logger.getLogger("appraisal").info("Random forest - avgPercentError: " + randomForestSImputationResult40.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: Random forest - 50% absence")
      Logger.getLogger("appraisal").info("Random forest - totalError: " + randomForestSImputationResult50.totalError)
      Logger.getLogger("appraisal").info("Random forest - avgPercentError: " + randomForestSImputationResult50.avgPercentError)

      Logger.getLogger("appraisal").info("===========================================================")

      Logger.getLogger("appraisal").info("Algoritmo: Multilayer perceptron - 10% absence")
      Logger.getLogger("appraisal").info("Multilayer perceptron - totalError: " + multilayerSImputationResult10.totalError)
      Logger.getLogger("appraisal").info("Multilayer perceptron - avgPercentError: " + multilayerSImputationResult10.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: Multilayer perceptron - 20% absence")
      Logger.getLogger("appraisal").info("Multilayer perceptron - totalError: " + multilayerSImputationResult20.totalError)
      Logger.getLogger("appraisal").info("Multilayer perceptron - avgPercentError: " + multilayerSImputationResult20.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: Multilayer perceptron - 30% absence")
      Logger.getLogger("appraisal").info("Multilayer perceptron - totalError: " + multilayerSImputationResult30.totalError)
      Logger.getLogger("appraisal").info("Multilayer perceptron - avgPercentError: " + multilayerSImputationResult30.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: Multilayer perceptron - 40% absence")
      Logger.getLogger("appraisal").info("Multilayer perceptron - totalError: " + multilayerSImputationResult40.totalError)
      Logger.getLogger("appraisal").info("Multilayer perceptron - avgPercentError: " + multilayerSImputationResult40.avgPercentError)

      Logger.getLogger("appraisal").info("-----------------------------------------------------------")

      Logger.getLogger("appraisal").info("Algoritmo: Multilayer perceptron - 50% absence")
      Logger.getLogger("appraisal").info("Multilayer perceptron - totalError: " + multilayerSImputationResult50.totalError)
      Logger.getLogger("appraisal").info("Multilayer perceptron - avgPercentError: " + multilayerSImputationResult50.avgPercentError)

    } catch {

      case ex: Exception => Logger.getLogger("appraisal").error(ex)

    }

  }

}