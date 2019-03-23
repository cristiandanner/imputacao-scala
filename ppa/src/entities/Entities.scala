package entities

import org.apache.spark.rdd._

object Entities {

  final case class CategoricResult(lineId: Long, originalValue: Option[String], imputationValue: String,
                                   error: Option[Int] = null)

  final case class NumericResult(lineId: Long, originalValue: Option[Double], imputationValue: Double,
                                 error: Option[Double] = null)

  final case class CategoricImputationResult(
    result:     RDD[CategoricResult],
    totalError: Option[Long]         = null, avgPercentError: Option[Double] = null)

  final case class NumericImputationResult(
    result:     RDD[NumericResult],
    totalError: Option[Double]     = null, avgPercentError: Option[Double] = null)

}