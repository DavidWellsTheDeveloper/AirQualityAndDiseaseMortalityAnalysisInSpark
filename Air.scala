import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark._
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SQLContext

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }

import org.apache.spark.ml.{ Pipeline, PipelineStage }
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.classification.RandomForestClassificationModel

object Air {
  case class Air (
    label: Double,
    gooddays: Double, moderatedays: Double, unhealthysensitivedays: Double, unhealthydays: Double,
    veryunhealthydays: Double, hazardousdays: Double, maxAQI: Double, ninetypercentileAQI: Double, medianAQI: Double,
    codays: Double, no2days: Double, ozonedays: Double, so2days: Double, pm25days: Double,
    pm10days: Double
  )
  
  def parseAir(line: Array[String]): Air = {
    Air (
      line(21).toDouble,
      line(5).toDouble, line(6).toDouble, line(7).toDouble, line(8).toDouble, line(9).toDouble,
      line(10).toDouble, line(11).toDouble, line(12).toDouble, line(13).toDouble, line(14).toDouble,
      line(15).toDouble, line(16).toDouble, line(17).toDouble, line(18).toDouble, line(19).toDouble
    )
  }
  
  def parseRDD(rdd: RDD[String]): RDD[Array[String]] = {
    rdd.map(_.split("\\t")).map(_.map(_.toString))
  }
  
  def main(args: Array[String]): Unit={

    val conf = new SparkConf().setAppName("airqmort").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext._
    import sqlContext.implicits._
    
    val rdd = sc.textFile(args(0))
    val newrdd = rdd.mapPartitionsWithIndex {
      (idx, iter) => if (idx == 0) iter.drop(1) else iter 
    }

    val airDF = parseRDD(newrdd).map(parseAir).toDF().cache()
    airDF.registerTempTable("air")
    airDF.printSchema

    airDF.show

    sqlContext.sql("SELECT label, avg(gooddays) as avggood, avg(unhealthydays) as avgunhealthy, avg(maxAQI) as avgmaxaqi  FROM air GROUP BY label ").show

    airDF.describe("gooddays").show
    airDF.groupBy("label").avg("gooddays").show

    val featureCols = Array("gooddays", "moderatedays", "unhealthysensitivedays", "unhealthydays", "veryunhealthydays",
      "hazardousdays", "maxAQI", "ninetypercentileAQI", "medianAQI", "codays",
      "no2days", "ozonedays", "so2days", "pm25days", "pm10days"
    )
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(airDF)
    df2.show

    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("clabel")
    val df3 = labelIndexer.fit(df2).transform(df2)
    df3.show
    val splitSeed = 5043
    val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)

    val classifier = new RandomForestClassifier().setImpurity("gini").setMaxDepth(3).setNumTrees(20).setFeatureSubsetStrategy("auto").setSeed(5043)
    val model = classifier.fit(trainingData)

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("clabel")
    val predictions = model.transform(testData)
    model.toDebugString
    
    val importances = model.featureImportances
    
    val importantfeatures = importances.toArray.zipWithIndex
            .map(_.swap)
            .sortBy(-_._2)

    val accuracy = evaluator.evaluate(predictions)
    //println("accuracy before pipeline fitting" + accuracy)
    
    val rawaccp = "accuracy before pipeline fitting" + accuracy

    val rm = new RegressionMetrics(
      predictions.select("prediction", "clabel").rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    )
    
    val msep = "MSE: " + rm.meanSquaredError
    val maep = "MAE: " + rm.meanAbsoluteError
    val rmsep = "RMSE Squared: " + rm.rootMeanSquaredError
    val rsp = "R Squared: " + rm.r2
    val exvarp = "Explained Variance: " + rm.explainedVariance + "\n"
    
    /*println("MSE: " + rm.meanSquaredError)
    println("MAE: " + rm.meanAbsoluteError)
    println("RMSE Squared: " + rm.rootMeanSquaredError)
    println("R Squared: " + rm.r2)
    println("Explained Variance: " + rm.explainedVariance + "\n")*/

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxBins, Array(25, 31))
      .addGrid(classifier.maxDepth, Array(5, 10))
      .addGrid(classifier.numTrees, Array(20, 60))
      .addGrid(classifier.impurity, Array("entropy", "gini"))
      .build()

    val steps: Array[PipelineStage] = Array(classifier)
    val pipeline = new Pipeline().setStages(steps)

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    val pipelineFittedModel = cv.fit(trainingData)

    val predictions2 = pipelineFittedModel.transform(testData)
    val accuracy2 = evaluator.evaluate(predictions2)
    //println("accuracy after pipeline fitting" + accuracy2)
    
    val pipeaccp = "accuracy after pipeline fitting" + accuracy2
    
    val noidea = pipelineFittedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(0)

    //println(pipelineFittedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(0))

    pipelineFittedModel
      .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(0)
      .extractParamMap

    val rm2 = new RegressionMetrics(
      predictions2.select("prediction", "clabel").rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    )
    
    val ppmse = "MSE: " + rm2.meanSquaredError
    val ppmae = "MAE: " + rm2.meanAbsoluteError
    val pprmse = "RMSE Squared: " + rm2.rootMeanSquaredError
    val pprs = "R Squared: " + rm2.r2
    val ppexvar = "Explained Variance: " + rm2.explainedVariance + "\n"
    
    val pimportances = pipelineFittedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages.head.asInstanceOf[RandomForestClassificationModel].featureImportances
    
    val pimportantfeatures = pimportances.toArray.zipWithIndex
            .map(_.swap)
            .sortBy(-_._2)
    /*println("MAE: " + rm2.meanAbsoluteError)
    println("RMSE Squared: " + rm2.rootMeanSquaredError)
    println("R Squared: " + rm2.r2)
    println("Explained Variance: " + rm2.explainedVariance + "\n")*/
    
    val ll = List(rawaccp, msep, maep, rmsep, rsp, exvarp, pipeaccp, noidea, ppmse, ppmae, pprmse, pprs, ppexvar)
    val rpr = sc.parallelize(ll)
    val imf = sc.parallelize(pimportantfeatures)
    rpr.saveAsTextFile(args(1))
    imf.saveAsTextFile(args(2))
    
    /*val file = new File(output)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(pr)
    bw.close()*/
    
    /*val featureCols = Array("a", "b", "c", "d", "e")
    val featureImportance = Vectors.dense(Array(0.15, 0.25, 0.1, 0.35, 0.15)).toSparse
    val res = featureCols.zip(featureImportance.toArray).sortBy(-_._2)*/

  }
}