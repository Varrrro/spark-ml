from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier, LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def load_data(sc, path):
    sqlc = SQLContext(sc)
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(
        sc._jsc.hadoopConfiguration())
    if fs.exists(sc._jvm.org.apache.hadoop.fs.Path(path)):
        data = sqlc.read.csv(path, header=True, inferSchema=True)
    else:
        headers = sc.textFile("/user/datasets/ecbdl14/ECBDL14_IR2.header") \
            .filter(lambda line: "@inputs" in line or "@outputs" in line) \
            .flatMap(lambda line: line.replace(",", "").split()) \
            .filter(lambda word: word != "@inputs" and word != "@outputs") \
            .collect()

        data = sqlc.read.csv(
            "/user/datasets/ecbdl14/ECBDL14_IR2.data", header=False, inferSchema=True)

        for i, colname in enumerate(data.columns):
            data = data.withColumnRenamed(colname, headers[i])

        data = data.select("PSSM_r2_-3_S", "PSSM_central_0_P", "PSSM_r1_1_Y",
                           "PSSM_r1_-3_R", "PSSM_r1_0_R", "PredSS_central_2", "class")

        data.write.csv(path, header=True)

    return data


def prepare_data(data):
    # Balance classes by undersampling
    positives = data.filter(data["class"] == 1)
    negatives = data.filter(data["class"] == 0)
    ratio = float(positives.count()) / float(data.count())
    sampled_negatives = negatives.sample(False, ratio)
    data = positives.union(sampled_negatives)

    # Split into train and test datasets
    train, test = data.randomSplit([0.7, 0.3])

    return train, test


def create_preprocess_pipeline():
    si = StringIndexer(
        inputCol="PredSS_central_2",
        outputCol="PredSS_central_2_indexed"
    )

    va = VectorAssembler(
        inputCols=["PSSM_r2_-3_S", "PSSM_central_0_P", "PSSM_r1_1_Y",
                   "PSSM_r1_-3_R", "PSSM_r1_0_R", "PredSS_central_2_indexed"],
        outputCol="features"
    )

    return Pipeline(stages=[si, va])


def train_random_forest(prepro, train):
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="class",
    )

    rf_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 50, 100]) \
        .addGrid(rf.impurity, ["gini", "entropy"]) \
        .build()

    rf_cv = CrossValidator(
        estimator=Pipeline(stages=[prepro, rf]),
        estimatorParamMaps=rf_grid,
        evaluator=BinaryClassificationEvaluator(labelCol="class"),
        numFolds=3
    )

    return rf_cv.fit(train)


def train_multilayer_perceptron(prepro, train):
    mlp = MultilayerPerceptronClassifier(
        featuresCol="features",
        labelCol="class",
        predictionCol="rawPrediction",
        maxIter=100
    )

    mlp_grid = ParamGridBuilder() \
        .addGrid(mlp.layers, [[6, 4, 2], [6, 8, 4, 2], [6, 8, 2]]) \
        .build()

    mlp_cv = CrossValidator(
        estimator=Pipeline(stages=[prepro, mlp]),
        estimatorParamMaps=mlp_grid,
        evaluator=BinaryClassificationEvaluator(labelCol="class"),
        numFolds=3
    )

    return mlp_cv.fit(train)


def train_logistic_regression(prepro, train):
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="class",
        maxIter=100,
        family="multinomial"
    )

    lr_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
        .addGrid(lr.elasticNetParam, [0.5, 0.6, 0.8]) \
        .build()

    lr_cv = CrossValidator(
        estimator=Pipeline(stages=[prepro, lr]),
        estimatorParamMaps=lr_grid,
        evaluator=BinaryClassificationEvaluator(labelCol="class"),
        numFolds=3
    )

    return lr_cv.fit(train)


def check_fit_params(sc, models):
    sqlc = SQLContext(sc)
    results_schema = StructType([
        StructField("classifier", StringType()),
        StructField("params", StringType()),
        StructField("auc", FloatType())
    ])
    results = sqlc.createDataFrame(sc.emptyRDD(), schema=results_schema)
    for classifier, model in models.items():
        for i, combination in enumerate(model.getEstimatorParamMaps()):
            params = ["%s: %s" % (p.name, str(v))
                      for p, v in combination.items()]

            param_results = sc.parallelize(
                [(classifier, "-".join(params), model.avgMetrics[i])])
            param_results = sqlc.createDataFrame(
                param_results, schema=results_schema)

            results = results.union(param_results)

    results.coalesce(1).write.csv("fit-metrics", header=True)


def evaluate(sc, models, test):
    sqlc = SQLContext(sc)
    results_schema = StructType([
        StructField("classifier", StringType()),
        StructField("auc", FloatType())
    ])
    results = sqlc.createDataFrame(sc.emptyRDD(), schema=results_schema)
    for classifier, model in models.items():
        bce = BinaryClassificationEvaluator(labelCol="class")
        auc = bce.evaluate(model.transform(test))

        evaluation = sc.parallelize([(classifier, auc)])
        evaluation = sqlc.createDataFrame(evaluation, schema=results_schema)

        results = results.union(evaluation)

    results.coalesce(1).write.csv("test-metrics", header=True)


if __name__ == "__main__":
    conf = SparkConf().setAppName("Practica 4 - Victor Vazquez Rodriguez")
    sc = SparkContext(conf=conf)

    data = load_data(sc, "./filteredC.small.training")
    train, test = prepare_data(data)

    prepro = create_preprocess_pipeline()

    models = {}
    models["rf"] = train_random_forest(prepro, train)
    models["mlp"] = train_multilayer_perceptron(prepro, train)
    models["lr"] = train_logistic_regression(prepro, train)

    check_fit_params(sc, models)
    evaluate(sc, models, test)

    sc.stop()
