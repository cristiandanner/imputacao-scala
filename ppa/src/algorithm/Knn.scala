package algorithm

import scala.collection.mutable.HashMap

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.col

import utility.Utility

object Knn {

  def run(idf: DataFrame, k: Int, attribute: String, params: HashMap[String, Any] = null): entities.Entities.CategoricImputationResult = {
    
    val attributes: Array[String] = params("attributes").asInstanceOf[Array[String]]

    val removeCol = idf.columns.diff(attributes).filter(!_.equals("lineId"))
    val calcCol = attributes.filter(!_.equals(attribute))

    var fidf = utility.Utility.filterNullAndNonNumeric(idf.drop(removeCol: _*), calcCol)

    calcCol.foreach(att => fidf = fidf.withColumn(att, utility.Utility.toDouble(col(att))))

    fidf = fidf.withColumn("originalValue", col(attribute)).drop(attribute)

    val rdf = knn(fidf, k, calcCol).filter(_._2 == null)

    entities.Entities.CategoricImputationResult(rdf.map(r => entities.Entities.CategoricResult(r._1, r._2, r._3)))

  }

  def knn(fidf: DataFrame, k: Int, calcCol: Array[String]): RDD[(Long, Option[String], String)] = {

    val lIdIndex = fidf.columns.indexOf("lineId")
    val oValIndex = fidf.columns.indexOf("originalValue")

    val context = fidf.sparkSession.sparkContext
    val cidf = context.broadcast(fidf.filter(_.get(oValIndex) != null).toDF())

    val rdf: RDD[(Long, Option[String], String)] = fidf.rdd.map(row => {

      val lineId = row.getLong(lIdIndex)
      val originalValue: Option[String] = if (row.get(oValIndex) != null) Some(row.getString(oValIndex)) else null
      val imputationValue = if (row.get(oValIndex) != null) row.getString(oValIndex) else knn(row, cidf, k, calcCol)

      (
        lineId,
        originalValue,
        imputationValue)
    })

    rdf

  }

  def knn(row: Row, cidf: Broadcast[DataFrame], k: Int, calcCol: Array[String]): String = {

    val dist = Utility.euclidianDist(row, cidf, calcCol).sortBy(_._3)
    val context = cidf.value.sparkSession.sparkContext

    context.parallelize(dist.take(k)).map(x => (x._2, 1)).reduceByKey((x, y) => x + y).sortBy(_._2, false).map(_._1).take(1)(0)

  }

}