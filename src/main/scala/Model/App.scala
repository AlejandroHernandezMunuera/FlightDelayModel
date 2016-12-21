package Model

import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Column
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
import java.util._

import org.apache.avro.generic.GenericData.StringType
import org.apache.spark.sql.types.IntegerType

import scala.collection.mutable
/**
 * @author ${user.name}
 */
object App {

  val getConcatenated = udf( ( first: String, second: String, third: String ) =>
  {
    first + "-" + second + "-" + third + ":"
  } )

  val isLeap = udf( ( year: Int ) =>
  {
    year % 400 == 0 || ( ( year % 4 == 0 ) && ( year % 100 != 0 ) )
  } )

  val dayCount = Array( 0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 )

  val dayOfYear = udf ( ( isLeap: Boolean, numberMonth: Int, day: Int ) =>
  {
    if( isLeap && numberMonth > 2 )
      (dayCount( numberMonth ) + day + 1).toDouble
    else
      (dayCount( numberMonth ) + day).toDouble
  } )

  val weekOfYear = udf ( ( dayOfYear: Double) =>
  {
    (dayOfYear/7).toInt.toDouble
  } )

  val minuteOfDay = udf ( ( hhmm: Int ) =>
  {
    val aux = hhmm.toString
    if( aux.length == 4 )
      aux.slice( 0, 2 ).toDouble * 60 + aux.slice( 2, 4 ).toDouble
    else if( aux.length == 3 )
      aux.slice( 0, 1 ).toDouble * 60 + aux.slice( 1, 3 ).toDouble
    else
      aux.toDouble
  } )

  val roundHour = udf ( ( hhmm: Int ) =>
  {
    val aux = hhmm.toString
    if( aux.length == 1 || ( aux.length == 2 && aux.toInt < 30 ) )
      0.toDouble
    else if( aux.length == 2 && aux.toInt >= 30 )
      1.toDouble
    else if( aux.length == 3  )
      if( aux.slice( 1, 3 ).toInt >= 30 )
        aux.slice( 0, 1 ).toDouble + 1
      else
        aux.slice( 0, 1 ).toDouble
    else if(  aux.slice( 2, 4 ).toInt >= 30 )
      aux.slice( 0, 2 ).toDouble + 1
    else
      aux.slice( 0, 2 ).toDouble
  } )

  val toStringAux = udf[ String, Int ]( _.toString )

  def main( args : Array[ String ] )
  {
    //if(args.length!=1)
      //throw new Exception("Not correct number of arguments")
    Logger.getLogger( "org" ).setLevel( Level.OFF )
    Logger.getLogger( "akka" ).setLevel( Level.OFF )

    val spark = SparkSession.builder().appName( "test" ).config( "spark.master", "local" ).getOrCreate()

    val flights = spark
      .read
      .format( "com.databricks.spark.csv" )
      .option( "header", "true" )
      .option( "inferSchema", "true" )
      //.load( args(0) )
      .load( "/Users/alejandrohernandezmunuera/Documents/Master/UPM/1ºSemester/Big data/2008_80k.csv" )
      .withColumn("Year", column("Year").cast("double"))
      .withColumn("DayOfWeek", column("DayOfWeek").cast("double"))
      .withColumn("DepTime", column("DepTime").cast("double"))
      .withColumn("CRSElapsedTime", column("CRSElapsedTime").cast("double"))
      .withColumn("ArrDelay", column("ArrDelay").cast("double"))
      .withColumn("DepDelay", column("DepDelay").cast("double"))
      .withColumn("TaxiOut", column("TaxiOut").cast("double"))
      .withColumn("Distance", column("Distance").cast("double"))
      .withColumn("FlightNum", toStringAux(column("FlightNum")))

      .where( col( "Cancelled" ) === 0)
      .where( col( "DepTime" ).isNotNull )
      .where( col( "CRSElapsedTime" ).isNotNull )
      .where( col( "ArrDelay" ).isNotNull )
      .where( col( "DepDelay" ).isNotNull )
      .where( col( "Distance" ).isNotNull )

      .drop( "Cancelled", "CancellationCode", "ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted",
        "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay" )

      .withColumn("IsLeapYear", isLeap( col("Year") ) )
      .withColumn( "DayOfYear", dayOfYear(col("isLeapYear"),col("Month"),col("DayofMonth")) )
      .withColumn("WeekOfYear",weekOfYear(col("DayOfYear")))
      .drop("IsLeapYear","Month","DayOfMonth")

      //.withColumn("DepTimeMinOfDay",minuteOfDay(col("DepTime")))
      .withColumn("DepTimeHour",roundHour(col("DepTime")))

      //.withColumn("CRSDepTimeMinOfDay",minuteOfDay(col("CRSDepTime")))
      //.withColumn("CRSDepTimeHour",roundHour(col("CRSDepTime")))

      //.withColumn("CRSArrTimeMinOfDay",minuteOfDay(col("CRSArrTime")))
      .withColumn("CRSArrTimeHour",roundHour(col("CRSArrTime")))

      .drop("DepTime","CRSDepTime","CRSArrTime")

      .withColumn("CRSSpeed",col("Distance")/col("CRSElapsedTime"))
      .withColumn( "DateString", getConcatenated( col( "DayOfYear" ), col( "Year" ), toStringAux(column("CRSArrTimeHour")) ))



    val numberFlights = flights.groupBy( "Dest", "DateString" ).count().withColumnRenamed("count","NumberOfFlightsLandingSameDestTime")


    var finalDF = flights.join(numberFlights,flights("Dest") === numberFlights("Dest") &&
      flights("DateString") === numberFlights("DateString") , "left_outer").drop(numberFlights("Dest"))
      .drop(numberFlights("DateString"))
      .drop("DateString")

    import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

    finalDF.printSchema()
    finalDF.show()

    val aux = finalDF.where( col( "TaxiOut" ).isNotNull)
    val input = mutable.MutableList[String]("CRSElapsedTime", "DepDelay", "Distance", "CRSSpeed",
      "NumberOfFlightsLandingSameDestTime")

    if( aux.count != 0 && aux.count / finalDF.count > 0.7 )
      {
        finalDF = aux
        input += "TaxiOut"
      }
    else
      {
        finalDF = finalDF.drop("TaxiOut")
      }

    val columnNames = Array( "UniqueCarrier", "FlightNum", "TailNum","Origin", "Dest",
      "Year","DayOfWeek","DayOfYear", "WeekOfYear", "CRSArrTimeHour", "DepTimeHour"  )



    for(name<-columnNames)
    {
      if( finalDF.groupBy( name ).count().count()>1)
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
          finalDF = finalDF.drop(name)
          finalDF = finalDF.drop(name+"Index")
        }
      else
        finalDF = finalDF.drop(name)

    }



    finalDF.printSchema()
    finalDF.show()
    println(input)
    import org.apache.spark.ml.regression.LinearRegression
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.Pipeline


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
    lrModel.transform(test).show

    import org.apache.spark.ml.evaluation.RegressionEvaluator
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
