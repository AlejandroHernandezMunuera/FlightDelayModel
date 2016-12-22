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

  //getConcatenated: Concatenate 3 strings with "-" between them
  val getConcatenated = udf( ( first: String, second: String, third: String ) =>
  {
    first + "-" + second + "-" + third
  } )

  //isLeap: Check if the atrribute year is leap
  val isLeap = udf( ( year: Int ) =>
  {
    year % 400 == 0 || ( ( year % 4 == 0 ) && ( year % 100 != 0 ) )
  } )

  //dayCount: each month has a position in the Array. Its position is the day of the year of the first
  // day of the month(supposing the year is not leap)
  val dayCount = Array( 0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 )

  //dayOfYear: return the day of the year having as attribute if the year is leap, the month number and
  // the day of the month
  val dayOfYear = udf ( ( isLeap: Boolean, numberMonth: Int, day: Int ) =>
  {
    if( isLeap && numberMonth > 2 )
      dayCount( numberMonth ) + day + 1
    else
      dayCount( numberMonth ) + day
  } )

  //weekOfYear: return the weeek of the year having as attribute the day of the year
  val weekOfYear = udf ( ( dayOfYear: Int) =>
  {
    dayOfYear/7
  } )

  //weekOfMonth: return the week of the month having as attribute the day of the month
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

  //minuteOfDay: return the minute of the day having as attribute an integer with the
  // two first digits from the right being the minute and the others the hour
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

  //roundHour: return the round hour of the day having as attribute an integer with the
  // two first digits from the right being the minute and the others the hour
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

  //toStringAux: from integer to String
  val toStringAux = udf[ String, Int ]( _.toString )

  def main( args : Array[ String ] )
  {

    //remove the logs
    Logger.getLogger( "org" ).setLevel( Level.OFF )
    Logger.getLogger( "akka" ).setLevel( Level.OFF )

    //check the number of attributes
    //if(args.length!=1)
      //throw new Exception("Not correct number of arguments")

    //val spark = SparkSession.builder().enableHiveSupport().appName( "test" ).getOrCreate()
    val spark = SparkSession.builder().appName( "test" ).config( "spark.master", "local" ).getOrCreate()

    //read Data file and perform changes
    val flights = spark
      .read
      .format( "com.databricks.spark.csv" )
      .option( "header", "true" )
      .option( "inferSchema", "true" )
      .load( "/Users/alejandrohernandezmunuera/Documents/Master/UPM/1ºSemestre/Big data/2008_80k.csv" )
      //.load( args(0) )
      //cast columns in order to use them in the regression model
      .withColumn("DepTime", column("DepTime").cast(IntegerType))
      .withColumn("CRSElapsedTime", column("CRSElapsedTime").cast("double"))
      .withColumn("ArrDelay", column("ArrDelay").cast("double"))
      .withColumn("DepDelay", column("DepDelay").cast("double"))
      .withColumn("TaxiOut", column("TaxiOut").cast("double"))
      .withColumn("Distance", column("Distance").cast("double"))
      //filter cancelled flights and null values
      .where( col( "Cancelled" ) === 0)
      .where( col( "DepTime" ).isNotNull )
      .where( col( "CRSElapsedTime" ).isNotNull )
      .where( col( "ArrDelay" ).isNotNull )
      .where( col( "DepDelay" ).isNotNull )
      .where( col( "Distance" ).isNotNull )
      //drop not allowed variables and "Cancelled", "CancellationCode", "TailNum" due to not importance
      //in the model
      .drop( "Cancelled", "CancellationCode", "ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted",
        "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "TailNum" )
      //new Columns
      .withColumn("IsLeapYear", isLeap( col("Year") ) )
      .withColumn( "DayOfYear", dayOfYear(col("isLeapYear"),col("Month"),col("DayofMonth")) )
      .withColumn("WeekOfYear", weekOfYear(col("DayOfYear")))
      .withColumn("WeekOfMonth", weekOfMonth(col("DayofMonth")))
      .withColumn("DepTimeHour",roundHour(col("DepTime")))
      .withColumn("CRSArrTimeHour",roundHour(col("CRSArrTime")))
      .withColumn("CRSSpeed",col("Distance")/col("CRSElapsedTime"))
      .withColumn( "DateString", getConcatenated( col( "DayOfYear" ), col( "Year" ), toStringAux(column("CRSArrTimeHour")) ))

    //new DataFrame: flights grouped by date and destination and count how many times the values
    // of both variables are repeated
    val numberFlights = flights.groupBy( "Dest", "DateString" ).count()
      .withColumnRenamed("count","NumberOfFlightsLandingSameDestTime")

    //join tables to have get the variable "NumberOfFlightsLandingSameDestTime" in the original dataframe
    var finalDF = flights.join(numberFlights,flights("Dest") === numberFlights("Dest") &&
      flights("DateString") === numberFlights("DateString") , "left_outer").drop(numberFlights("Dest"))
      .drop(numberFlights("DateString"))

    //to check if there are rows with "TaxiOut" not null
    val aux = finalDF.where( col( "TaxiOut" ).isNotNull)

    //variables that will be used in the model
    val input = mutable.MutableList[String]("CRSElapsedTime","DepDelay", "CRSSpeed",
      "NumberOfFlightsLandingSameDestTime")

    //if there is more than 70% of rows with "TaxiOut" not null then filter and include the
    //variable in the model
    if( aux.count != 0 && aux.count / finalDF.count > 0.7 )
      {
        finalDF = aux
        input += "TaxiOut"
      }

    //Categorical variables to be analysed
    val columnNames = Array( "Year", "DayOfWeek", "UniqueCarrier","Origin", "Dest",
      "WeekOfMonth", "CRSArrTimeHour","DepTimeHour"  )

    for(name<-columnNames)
    {
      //if variable has more than one different value then create StringIndexer and HotEncoders
      //and add it to the model
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

    //split data in training(70%) data and test data(30%)
    val split = finalDF.randomSplit(Array(0.7,0.3))
    val training = split(0)
    val test = split(1)

    //set inputs variables and output variable for the model
    val assembler = new VectorAssembler()
      .setInputCols(input.toArray)
      .setOutputCol("ArrDelayNew")

    //create linear model regression
    val lr = new LinearRegression()
      .setFeaturesCol("ArrDelayNew")
      .setLabelCol("ArrDelay")
      .setMaxIter(10)
      .setElasticNetParam(0.8)

    //create pipeline
    val pipeline = new Pipeline()
      .setStages(Array(assembler, lr))

    //train model
    val lrModel = pipeline.fit(training)

    //test model
    val predictions = lrModel.transform(test)

    //evaluator of the model (R-squared)
    val eval_r1 = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setMetricName("r2")

    //evaluator of the model (mean squared error)
    val eval_r2 = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setMetricName("rmse")

    //print metrics
    println("R-squared: " + eval_r1.evaluate(predictions))
    println("rmse: "+ eval_r2.evaluate(predictions))

  }
}
