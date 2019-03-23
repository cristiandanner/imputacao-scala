package algorithm

import scala.collection.mutable.HashMap

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.DataFrame

object DecisionTree {

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

      (lineId, LabeledPoint(callback(attributeValue), Vectors.dense(values)))
    })
    
    val splits = vectorsRdd.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    
    // Train a DecisionTree model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = params("numClasses").asInstanceOf[Int]
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = params("impurity").asInstanceOf[String]
    val maxDepth = params("maxDepth").asInstanceOf[Int]
    val maxBins = params("maxBins").asInstanceOf[Int]

    val decisionTreeModel = org.apache.spark.mllib.tree.DecisionTree.trainClassifier(trainingData.map(_._2), numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.filter(_._2.label == 0).map { point =>
      val prediction = decisionTreeModel.predict(point._2.features)
      (point._1, null, prediction)
    }

    entities.Entities.NumericImputationResult(labelAndPreds.map(r => entities.Entities.NumericResult(r._1, r._2, r._3)))

  }

}