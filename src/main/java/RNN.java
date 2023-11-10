import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;

public class RNN {
    public static void rnn(int nEpoches)throws Exception{
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new FileSplit(new File("data\\rnn_train.csv")));

        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new FileSplit(new File("data\\rnn_test.csv")));


        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(
                trainFeatures,
                1,
                2,
                1,
                false
        );

        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(
                trainFeatures,
                1,
                2,
                1,
                false
        );

        int numCharactersInAlphabet = 26;  // 字母表大小
        int embeddingSize = 10;  // 词嵌入维度
        int lstmLayerSize = 256;  // LSTM 层大小
        int numLabelClasses = 2;  // 分类的数量

        MultiLayerNetwork net;
        net = new MultiLayerNetwork(
                new NeuralNetConfiguration.Builder()
                        .seed(123)
                        .updater(new Adam(0.001))
                        .list()
                        .layer(0, new EmbeddingLayer.Builder()
                                .nIn(numCharactersInAlphabet)
                                .nOut(embeddingSize)
                                .build())
                        .layer(1, new LSTM.Builder()
                                .nIn(embeddingSize)
                                .nOut(lstmLayerSize)
                                .activation(Activation.RELU)
                                .build())
                        .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX)
                                .nIn(lstmLayerSize)
                                .nOut(numLabelClasses)
                                .build())
                        .setInputType(InputType.recurrent(embeddingSize))
                        .build()
        );
        net.init();

        for(int i=0;i<nEpoches;i++){
            net.fit(trainData);
        }

    }

    public static void main(String[] args) throws Exception {
        rnn(10);
    }

}
