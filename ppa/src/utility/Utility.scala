package utility

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf

object Utility {

  val to_String = udf[Option[String], String](x => if (x != null) Some(x.toString) else None)

  val toDouble = udf[Option[Double], String](x => if (x != null) Some(x.toDouble) else None)

  val toLong = udf[Long, String](_.toLong)

  def isNumeric(str: String): Boolean = str.matches("[-+]?\\d+(\\.\\d+)?")

  def loadDataset(spark: SparkSession, datasetName: String): DataFrame = {

    spark.read.option("header", true).csv(datasetName)

  }

  def euclidianDist(row: Row, rowc: Row, cColPos: Array[Int]): Double = {

    var dist = 0d

    cColPos.foreach(attIndex => {

      dist += scala.math.pow(row.getDouble(attIndex) - rowc.getDouble(attIndex), 2)

    })

    return scala.math.sqrt(dist)

  }

  def euclidianDist(row: Row, cidf: Broadcast[DataFrame], calcCol: Array[String]): RDD[(Long, String, Double)] = {

    val ecidf = cidf.value

    val lIdIndex = ecidf.columns.indexOf("lineId")
    val oValIndex = ecidf.columns.indexOf("originalValue")
    val cColPos = calcCol.map(ecidf.columns.indexOf(_))

    ecidf.rdd.map(rowc => (rowc.getLong(lIdIndex), rowc.getString(oValIndex), euclidianDist(row, rowc, cColPos)))

  }

  def filterNullAndNonNumeric(df: DataFrame, columns: Array[String] = null): DataFrame = {

    var _columns = df.columns
    if (columns != null)
      _columns = columns

    var rdf = df

    _columns.foreach(column => {

      val columnIndex = rdf.columns.indexOf(column)
      rdf = filterNullAndNonNumericByAtt(rdf, columnIndex)

    })

    rdf

  }

  def filterNullAndNonNumericByAtt(df: DataFrame, attIndex: Int): DataFrame = {

    df.filter(r => r.get(attIndex) != null && isNumeric(r.get(attIndex).toString()))

  }

}