package statistic

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType

object Statistic {

  def statisticInfo(ods: DataFrame, attribute: String, impres: entities.Entities.CategoricImputationResult): entities.Entities.CategoricImputationResult = {

    val lods = ods.withColumn("lineId", monotonically_increasing_id)

    val impds = impres.result.map(r => Row(r.lineId, r.originalValue, r.imputationValue))

    val schema = StructType(List(
      StructField("lineId", LongType, nullable = true),
      StructField("originalValue", StringType, nullable = true),
      StructField("imputationValue", StringType, nullable = true)))

    val impdf = ods.sqlContext.createDataFrame(impds, schema)

    impdf.createOrReplaceTempView("result")
    lods.createOrReplaceTempView("original")

    val result = impdf.sqlContext.sql("select r.lineid, o." + attribute + " as originalValue, r.imputationValue " +
      "from result r inner join original o on r.lineid = o.lineid order by r.lineid")
      .withColumn("originalValue", utility.Utility.to_String(col("originalValue"))).rdd

    val impResult = entities.Entities.CategoricImputationResult(result.map(r => {

      val isEqual = r.getString(2).equals(r.getString(1))
      var error = 0
      if (!isEqual)
        error += 1

      entities.Entities.CategoricResult(r.getLong(0), Some(r.getString(1)), r.getString(2), Some(error))

    }))

    val countTotal = impResult.result.count().doubleValue()
    val countError = impResult.result.filter(_.error == Some(1)).count()
    val avgPercentualError = (countError * 100) / countTotal

    entities.Entities.CategoricImputationResult(impResult.result, Some(countError), Some(avgPercentualError))

  }

  def numericStatisticInfo(ods: DataFrame, attribute: String, impres: entities.Entities.NumericImputationResult): entities.Entities.NumericImputationResult = {

    val lods = ods.withColumn("lineId", monotonically_increasing_id)

    val impds = impres.result.map(r => Row(r.lineId, r.originalValue, r.imputationValue))

    val schema = StructType(List(
      StructField("lineId", LongType, nullable = true),
      StructField("originalValue", DoubleType, nullable = true),
      StructField("imputationValue", DoubleType, nullable = true)))

    val impdf = ods.sqlContext.createDataFrame(impds, schema)

    impdf.createOrReplaceTempView("result")
    lods.createOrReplaceTempView("original")

    val result = impdf.sqlContext.sql("select r.lineid, o." + attribute + " as originalValue, r.imputationValue " +
      "from result r inner join original o on r.lineid = o.lineid order by r.lineid")
      .withColumn("originalValue", utility.Utility.toDouble(col("originalValue"))).rdd

    val impResult = entities.Entities.NumericImputationResult(result.map(r => {

      val isEqual = r.getDouble(2) == r.getDouble(1)
      var error = 0
      if (!isEqual)
        error += 1

      entities.Entities.NumericResult(r.getLong(0), Some(r.getDouble(1)), r.getDouble(2), Some(error))

    }))

    val countTotal = impResult.result.count().doubleValue()
    val countError = impResult.result.filter(_.error == Some(1)).count()
    val avgPercentualError = (countError * 100) / countTotal

    entities.Entities.NumericImputationResult(impResult.result, Some(countError), Some(avgPercentualError))

  }

}