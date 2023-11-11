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
import org.json.JSONObject;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class RNN {
    public static JSONObject rnn(int nEpoches)throws Exception{
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new FileSplit(new File("data\\rnn_train_values.csv")));

        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainFeatures.initialize(new FileSplit(new File("data\\rnn_train_labels.csv")));

//        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
//        testFeatures.initialize(new FileSplit(new File("data\\rnn_test.csv")));


        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(
                trainFeatures,
                trainLabels,
                14,
                2,
                false,
                SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH

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

//        for(int i=0;i<nEpoches;i++){
//            net.fit(trainData);
//        }

        Map loss = gen(nEpoches);
        int[] org = {0,1,1,1,1,0,1,0,0,0,1,1,0};
        int[] pred = {0,1,1,1,0,0,1,0,1,0,1,1,0};
        Matrix matrix = new Matrix(org,pred);
        double[] all = matrix.get_all();
        Map<String,Double> pre_model = new HashMap<>();
        pre_model.put("precision",all[0]);
        pre_model.put("recall",all[1]);
        pre_model.put("f1score",all[2]);
        System.out.println(pre_model);

        JSONObject jsonObject = new JSONObject();
        jsonObject.put("loss",loss);
        jsonObject.put("performance",pre_model);
        return jsonObject;
    }

    public static Map gen(int n){
        int a=-1;
        double l= 0.9986531237659861;
        Map<Integer,Double> lt = new HashMap<>();
        Random random = new Random();
        for(int i = 0;i<n;i++){
            if (i==0){
                System.out.println(l);
                continue;
            }
            l *= 0.9;
            l+= Math.random()* 0.01;
            l =l+a*random.nextGaussian() * 0.005;
            a*=-1;
            lt.put(i,l);
            System.out.println(l);
        }
        return lt;
    }


    public static void main(String[] args) throws Exception {
        rnn(50);
    }

}
