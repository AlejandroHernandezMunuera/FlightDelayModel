package Model

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.log4j.Logger
import org.apache.log4j.Level

import java.util._
/**
 * @author ${user.name}
 */
object App {

  def main( args : Array[ String ] )
  {
    Logger.getLogger( "org" ).setLevel( Level.OFF )
    Logger.getLogger( "akka" ).setLevel( Level.OFF )


    val spark = SparkSession.builder().appName( "test" ).config("spark.master", "local").getOrCreate()
    import spark.implicits._


    val sqlContext = new SQLContext( spark.sparkContext )
    import sqlContext.implicits._

    val getConcatenated = udf( (first: String, second: String, third: String ) => { first + "-" + second + "-" + third} )
    //val toInt = udf[ Int, String ]( _.toInt )
    //val toDouble = udf[ Double, String ]( _.toDouble )

    val flights = spark
      .read
      .format( "com.databricks.spark.csv" )
      .option( "header", "true" )
      .option( "inferSchema", "true" )
      .load( "/Users/alejandrohernandezmunuera/Documents/Master/UPM/1ºSemestre/Big data/2008.csv" )
      .drop( "ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn",
        "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay",
        "SecurityDelay", "LateAircraftDelay" )
      .withColumn( "DateString", getConcatenated( col( "DayOfMonth" ), col( "Month" ), col( "Year" ) ) )
      .withColumn( "DateTimeStampp", unix_timestamp( col( "DateString" ), "dd-MM-yyyy" ).cast( "timestamp" ) )
      .withColumn( "IsWeekend", col( "DayOfWeek" ) > 5 )
    //.withColumn( "DateTimeStampp", to_date( unix_timestamp( col( "Date" ), "dd-MM-yyyy" ).cast( "timestamp" ) ) )
    //.filter( unix_timestamp( col( "Date" ), "dd-MM-yyyy" ).cast( "timestamp" ) <= "1993-01-15" )
    flights.show( 100 )
    flights.printSchema()


    flights.groupBy(col("Origin"), col("DateString")).count().show
  }

}
