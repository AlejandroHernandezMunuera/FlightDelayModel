package Model

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator

import scala.collection.mutable
/**
 * @author ${Alejandro Hernandez, Daan Knoors}
 */
object App {

  val getConcatenated = udf( ( first: String, second: String, third: String ) =>
  {
    first + "-" + second + "-" + third
  } )

  val isLeap = udf( ( year: Int ) =>
  {
    year % 400 == 0 || ( ( year % 4 == 0 ) && ( year % 100 != 0 ) )
  } )

  val dayCount = Array( 0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 )

  val dayOfYear = udf ( ( isLeap: Boolean, numberMonth: Int, day: Int ) =>
  {
    if( isLeap && numberMonth > 2 )
      dayCount( numberMonth ) + day + 1
    else
      dayCount( numberMonth ) + day
  } )

  val weekOfYear = udf ( ( dayOfYear: Int) =>
  {
    dayOfYear/7
  } )

  val weekOfMonth = udf ( ( dayOfMonth: Int) =>
  {
    if( dayOfMonth <= 7 )
      1
    else if( dayOfMonth <= 14 )
      2
    else if( dayOfMonth <= 21)
      3
    else
      4
  } )

  val minuteOfDay = udf ( ( hhmm: Int ) =>
  {
    val aux = hhmm.toString
    if( aux.length == 4 )
      aux.slice( 0, 2 ).toInt * 60 + aux.slice( 2, 4 ).toInt
    else if( aux.length == 3 )
      aux.slice( 0, 1 ).toInt * 60 + aux.slice( 1, 3 ).toInt
    else
      aux.toInt
  } )

  val roundHour = udf ( ( hhmm: Int ) =>
  {
    val aux = hhmm.toString
    if( aux.length == 1 || ( aux.length == 2 && aux.toInt < 30 ) )
      0
    else if( aux.length == 2 && aux.toInt >= 30 )
      1
    else if( aux.length == 3  )
      if( aux.slice( 1, 3 ).toInt >= 30 )
        aux.slice( 0, 1 ).toInt + 1
      else
        aux.slice( 0, 1 ).toInt
    else if(  aux.slice( 2, 4 ).toInt >= 30 )
      aux.slice( 0, 2 ).toInt + 1
    else
      aux.slice( 0, 2 ).toInt
  } )

  val toStringAux = udf[ String, Int ]( _.toString )

  def main( args : Array[ String ] ) {

    Logger.getLogger( "org" ).setLevel( Level.OFF )
    Logger.getLogger( "akka" ).setLevel( Level.OFF )

    //if(args.length!=1)
      //throw new Exception("Not correct number of arguments")

    val spark = SparkSession.builder().appName( "test" ).config( "spark.master", "local" ).getOrCreate()
    //val spark = SparkSession.builder().enableHiveSupport().appName( "test" ).getOrCreate()

    val flights = spark
      .read
      .format( "com.databricks.spark.csv" )
      .option( "header", "true" )
      .option( "inferSchema", "true" )
      //.load( args(0) )
      .load( "/Users/alejandrohernandezmunuera/Documents/Master/UPM/1ºSemestre/Big data/2008_80k.csv" )
      .withColumn("DepTime", column("DepTime").cast(IntegerType))
      .withColumn("CRSElapsedTime", column("CRSElapsedTime").cast("double"))
      .withColumn("ArrDelay", column("ArrDelay").cast("double"))
      .withColumn("DepDelay", column("DepDelay").cast("double"))
      .withColumn("TaxiOut", column("TaxiOut").cast("double"))
      .withColumn("Distance", column("Distance").cast("double"))
      .where( col( "Cancelled" ) === 0)
      .where( col( "DepTime" ).isNotNull )
      .where( col( "CRSElapsedTime" ).isNotNull )
      .where( col( "ArrDelay" ).isNotNull )
      .where( col( "DepDelay" ).isNotNull )
      .where( col( "Distance" ).isNotNull )
      .drop( "Cancelled", "CancellationCode", "ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted",
        "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "TailNum" )
      .withColumn("IsLeapYear", isLeap( col("Year") ) )
      .withColumn( "DayOfYear", dayOfYear(col("isLeapYear"),col("Month"),col("DayofMonth")) )
      .withColumn("WeekOfYear", weekOfYear(col("DayOfYear")))
      .withColumn("WeekOfMonth", weekOfMonth(col("DayofMonth")))
      .withColumn("DepTimeHour",roundHour(col("DepTime")))
      .withColumn("CRSArrTimeHour",roundHour(col("CRSArrTime")))
      .withColumn("CRSSpeed",col("Distance")/col("CRSElapsedTime"))
      .withColumn( "DateString", getConcatenated( col( "DayOfYear" ), col( "Year" ), toStringAux(column("CRSArrTimeHour")) ))

    val numberFlights = flights.groupBy( "Dest", "DateString" ).count().withColumnRenamed("count","NumberOfFlightsLandingSameDestTime")

    var finalDF = flights.join(numberFlights,flights("Dest") === numberFlights("Dest") &&
      flights("DateString") === numberFlights("DateString") , "left_outer").drop(numberFlights("Dest"))
      .drop(numberFlights("DateString"))

    val aux = finalDF.where( col( "TaxiOut" ).isNotNull)

    val input = mutable.MutableList[String]("CRSElapsedTime","DepDelay", "CRSSpeed",
      "NumberOfFlightsLandingSameDestTime")

    if( aux.count != 0 && aux.count / finalDF.count > 0.7 )
      {
        finalDF = aux
        input += "TaxiOut"
      }

    val columnNames = Array( "Year", "DayOfWeek", "UniqueCarrier","Origin", "Dest",
      "WeekOfMonth", "CRSArrTimeHour","DepTimeHour"  )

    for(name<-columnNames)
    {
      if( finalDF.select(name).distinct().count()>1)
        {
          finalDF = new StringIndexer()
            .setInputCol(name)
            .setOutputCol(name+"Index")
            .fit(finalDF).transform(finalDF)
          finalDF = new OneHotEncoder()
            .setInputCol(name+"Index")
            .setOutputCol(name+"Vector")
            .transform(finalDF)
          input += name+"Vector"
        }
    }

    val split = finalDF.randomSplit(Array(0.7,0.3))
    val training = split(0)
    val test = split(1)

    val assembler = new VectorAssembler()
      .setInputCols(input.toArray)
      .setOutputCol("ArrDelayNew")

    val lr = new LinearRegression()
      .setFeaturesCol("ArrDelayNew")
      .setLabelCol("ArrDelay")
      .setMaxIter(10)
      .setElasticNetParam(0.8)

    val pipeline = new Pipeline()
      .setStages(Array(assembler, lr))

    val lrModel = pipeline.fit(training)

    val predictions = lrModel.transform(test)

    val eval_r1 = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setMetricName("r2")

    val eval_r2 = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setMetricName("rmse")

    val r2 = eval_r1.evaluate(predictions)

    val rmse = eval_r2.evaluate(predictions)

    println("R-squared: " + r2)
    println("rmse: "+ rmse)

  }
}
