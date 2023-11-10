import au.com.bytecode.opencsv.CSVParser;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
//import org.deeplearning4j.examples.utils.DownloaderUtility;
//import org.deeplearning4j.examples.utils.PlotUtil;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import jdk.jfr.consumer.RecordedClass;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;

import java.io.File;

public class CNN {
    public static void CNNmodel(double learningRate, int batchSize, int nEpochs, int numHiddenNodes) throws Exception {
        int seed = 123;//随机种子
        int nChannels = 1;
//        int numRows = 28;
//        int numColNums = 28;
        int outputNum = 10;

        String trainpath = "data\\mnist_train.csv";
        String testpath = "data\\mnist_test.csv";

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(trainpath)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, outputNum);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(testpath)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, outputNum);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam())
                .l2(learningRate*0.005)
                .list()
                .layer(0,new ConvolutionLayer.Builder(5,5)//卷积核大小
                        .nIn(nChannels)
                        .stride(1,1)
                        .nOut(numHiddenNodes)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2,new DenseLayer.Builder()
                        .nIn(numHiddenNodes)
                        .activation(Activation.RELU)
                        .nOut(500)
                        .build())
                .layer(3,new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(numHiddenNodes)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
//        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates

//        System.out.println("/////////");
        for(int i = 0;i<nEpochs;i++) {
            model.fit(trainIter);
            double score = model.score();
//            System.out.println("1111");
            System.out.println(score);
//            System.out.println("2222");
//            loss.put(index,score);
//            index=index+1;
        }

//        model.fit(trainIter, nEpochs);

        System.out.println("Evaluate model....");
        Evaluation eval = model.evaluate(testIter);
        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);
            eval.eval(labels, predicted);
        }
        //An alternate way to do the above loop
        //Evaluation evalResults = model.evaluate(testIter);

        //Print the evaluation statistics
        System.out.println(eval.stats());

        System.out.println("\n****************Example finished********************");
        //Training is complete. Code that follows is for plotting the data & predictions only
    }

    public static void main(String[] args) throws Exception {
        CNNmodel(0.001,64,10,50);//以上数据均由前端提供
    }

}
