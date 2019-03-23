package algorithm

import scala.collection.mutable.HashMap

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame

object MultilayerPerceptron {

  def run(idf: DataFrame, attribute: String, params: HashMap[String, Any] = null, callback: String => Double): entities.Entities.NumericImputationResult = {

    val attributes: Array[String] = params("attributes").asInstanceOf[Array[String]]

    val removeCol = idf.columns.diff(attributes).filter(_ != "lineId")
    val remidf = idf.drop(removeCol: _*)

    val context = remidf.sparkSession.sparkContext

    val calcCol = attributes.filter(_ != attribute)

    var fidf = context.broadcast(utility.Utility.filterNullAndNonNumeric(remidf, calcCol))

    val columns = fidf.value.columns

    val lineIdIndex = columns.indexOf("lineId")
    val attributeIndex = columns.indexOf(attribute)

    val vectorsRdd = fidf.value.rdd.map(row => {

      val lineId = row.getLong(lineIdIndex)
      val attributeValue = row.getString(attributeIndex)

      var values = new Array[Double](calcCol.length)

      for (i <- 0 to (calcCol.length - 1))
        values(i) = row.getString(columns.indexOf(calcCol(i))).toDouble

      (lineId, callback(attributeValue), Vectors.dense(values))
    })
    
    val splits = vectorsRdd.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    val spark = fidf.value.sparkSession
    val mdfTrain = spark.createDataFrame(train).toDF("lineId", "label", "features")
    val mdfTest = spark.createDataFrame(test).toDF("lineId", "label", "features")

    // specify layers for the neural network:
    // input layer of size 4 (features), and output of size 4 (classes)
    val layers = Array[Int](4, 5, 4)

    val mlp = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(params("blockSize").asInstanceOf[Int])
      .setSeed(params("seed").asInstanceOf[Long])
      .setMaxIter(params("maxIter").asInstanceOf[Int])

    val mlpModel = mlp.fit(mdfTrain)

    val result = mlpModel.transform(mdfTest)

    val predictions = result.select("lineId", "label", "prediction").where("label == 0")

    val lIdIndex = predictions.columns.indexOf("lineId")
    val lblIndex = predictions.columns.indexOf("label")
    val iValIndex = predictions.columns.indexOf("prediction")

    val labelAndPreds = predictions.rdd.map { row =>
      val lineId = row.getLong(lIdIndex)
      val imputationValue = row.getDouble(iValIndex)

      (lineId, null, imputationValue)
    }

    entities.Entities.NumericImputationResult(labelAndPreds.map(r => entities.Entities.NumericResult(r._1, r._2, r._3)))

  }

}