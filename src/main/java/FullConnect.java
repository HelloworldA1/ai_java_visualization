import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.json.JSONObject;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FullConnect {
    public static JSONObject FullConnect(double learningRate, int batchSize, int nEpochs, int numHiddenNodes) throws Exception {
        int seed = 123;//随机种子

        String trainPtah = "data\\flower_labels_noname.csv";
        double trainRatio = 0.9;

        MyDataset myDataset = new MyDataset(trainPtah);
        int numlabel = myDataset.getNuminput();//标签类别数
        int LabelIndex = myDataset.getLabelIndex();//标签索引
        int numfeature = LabelIndex;//特征数

        String[] path = splitData(trainPtah,trainRatio);
        String trainpath = path[0];
        String testpath = path[1];


        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(trainpath)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, LabelIndex, numlabel);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(testpath)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, LabelIndex, numlabel);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                .l2(learningRate * 0.005) // regularize learning model
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(numfeature)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(1,new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(2,new OutputLayer.Builder()
                        .nIn(64)
                        .nOut(numlabel)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
//        model.setListeners(new ScoreIterationListener(10)); //Print score every 10 parameter updates

        Map<Integer,Double> loss = new HashMap<>();
        int index = 0;
        for(int i = 0;i<nEpochs;i++) {
            model.fit(trainIter);
            double score = model.score();
            loss.put(index,score);
            index=index+1;
        }
//        System.out.println(loss.toString());

        Evaluation eval = new Evaluation(numlabel);
        System.out.println(numfeature);
        RegressionEvaluation eval_reg = new RegressionEvaluation(3);

        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);
            eval.eval(labels, predicted);
            eval_reg.eval(labels,predicted);
        }
        //An alternate way to do the above loop
//        Evaluation evalResults = model.evaluate(testIter);

        //Print the evaluation statistics
        Map<String,Double> performance = new HashMap<>();
        String evalStats = eval.stats();
        String[] lines = evalStats.split("\n");
        for (String line : lines) {
            if (line.contains("Accuracy")) {//通过率或准确率
                String[] text = line.split("[:,(\t]");
                 performance.put ("Accuracy",Double.parseDouble(text[1].replaceAll("^\\s+", "")));
            }else if (line.contains("Precision")) {
                String[] text = line.split("[:,(\t]");
                performance.put ("Precision",Double.parseDouble(text[1].replaceAll("^\\s+", "")));
            }else if(line.contains("Recall")){
                String[] text = line.split("[:,(\t]");
                performance.put ("Recall",Double.parseDouble(text[1].replaceAll("^\\s+", "")));
            }else if(line.contains("F1 Score")){
                String[] text = line.split("[:,(\t]");
                performance.put ("F1 Score",Double.parseDouble(text[1].replaceAll("^\\s+", "")));
                break;
            }
        }

        String out = eval_reg.stats();
        String[] outs = out.split("\n");
        for(String line:outs){
            if(line.contains("Column")){
                continue;
            }
            else{
                String[] out_line = line.split("    ");
                performance.put ("MSE",Double.parseDouble(out_line[1].replaceAll("^\\s+", "")));
                performance.put ("MAE",Double.parseDouble(out_line[2].replaceAll("^\\s+", "")));
                performance.put ("RMSE",Double.parseDouble(out_line[3].replaceAll("^\\s+", "")));
                performance.put ("RSE",Double.parseDouble(out_line[4].replaceAll("^\\s+", "")));
                performance.put ("R^2",Double.parseDouble(out_line[5].replaceAll("^\\s+", "")));
                break;
            }
        }

        System.out.println(performance.toString());



        JSONObject jsonObject = new JSONObject();
        jsonObject.put("loss",loss);
        jsonObject.put("performance",performance);
        return jsonObject;

    }

    public static String[] splitData(String dataFilePath, double trainRatio) throws IOException, InterruptedException {
        // 读取CSV文件
        FileReader fileReader = new FileReader(dataFilePath);
        Iterable<CSVRecord> records = CSVFormat.DEFAULT.parse(fileReader);
        //将数据加载到列表中
        List<CSVRecord> dataList = new ArrayList<>();
        for (CSVRecord record : records) {
            dataList.add(record);
        }

        int totalRows = dataList.size();
        int trainRows = (int) (totalRows * trainRatio);
        int testRows = totalRows - trainRows;

        // 写入训练数据
        String trainPath = getTrainFilePath();
        FileWriter trainFileWriter = new FileWriter(trainPath);
        CSVPrinter trainCsvPrinter = new CSVPrinter(trainFileWriter,CSVFormat.DEFAULT);

        for(int i=0;i<trainRows;i++){
            trainCsvPrinter.printRecord(dataList.get(i));
        }

        // 写入测试数据
        String testPtah = getTestFilePath();
        FileWriter testFileWriter = new FileWriter(testPtah);
        CSVPrinter testCsvPrinter = new CSVPrinter(testFileWriter,CSVFormat.DEFAULT);

        for(int i=trainRows;i<totalRows;i++){
            testCsvPrinter.printRecord(dataList.get(i));
        }

        trainFileWriter.close();
        testCsvPrinter.close();

        String[] path = new String[2];
        path[0]=trainPath;
        path[1]=testPtah;
        return path;
    }

    public  static String getTrainFilePath(){
        // 获取当前时间戳
        long timestamp = System.currentTimeMillis();
        // 构建文件名，使用时间戳
        String fileName = "Intermediate_steps_file/train_" + timestamp + ".csv";
        return fileName;
    }

    public  static String getTestFilePath(){
        // 获取当前时间戳
        long timestamp = System.currentTimeMillis();
        // 构建文件名，使用时间戳
        String fileName = "Intermediate_steps_file/test_" + timestamp + ".csv";
        return fileName;
    }

    public static void main(String[] args) throws Exception {
//        Logger logger = LoggerFactory.getLogger("org.deeplearning4j");
//        logger.info("Setting log level to INFO");
//        System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "INFO");
        FullConnect(0.01,12,30,20);//数据全由前端提供
    }
}
