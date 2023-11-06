import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.gui.beans.DataSource;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
//import org.apache.poi.ss.usermodel.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;


public class RandomForest_K_means {
    //k-means
    public static void K_means(Instances datas,int NUmClusters){
        try{
            SimpleKMeans kMeans = new SimpleKMeans();
            kMeans.setPreserveInstancesOrder(true);
            kMeans.setNumClusters(NUmClusters); // 设置簇的数量
            kMeans.buildClusterer(datas); // 训练模型

            // 获取聚类中心
            Instances centroids = kMeans.getClusterCentroids();
            System.out.println("聚类中心:");
            System.out.println(centroids);

            // 获取簇内误差平方和
            double squaredErrors = kMeans.getSquaredError();
            System.out.println("簇内误差平方和:"+squaredErrors);

            for (int i = 0; i < datas.numInstances(); i++) {
                int cluster = kMeans.clusterInstance(datas.instance(i));
                System.out.println("Instance " + (i + 1) + " is in cluster " + cluster);
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }


    //randomforest
    public static void RandomForest(MyDataset myDataset,int NumTrees,int depth,int NumSeed,int NumAttribute)  {

        try {
            RandomForest randomForest = new RandomForest();
            randomForest.setOptions(weka.core.Utils.splitOptions("-I "+NumTrees));//数的个数，默认100
//            randomForest.setOptions(weka.core.Utils.splitOptions("-depth "+depth));//最大树深度，默认0，即不限制
            randomForest.setOptions(weka.core.Utils.splitOptions("-S "+NumSeed));//随机种子参数，默认1
            randomForest.setOptions(weka.core.Utils.splitOptions("-K "+NumAttribute));//特征子采样参数，默认0
            randomForest.buildClassifier(myDataset.trainset);

            Evaluation eval = new Evaluation(myDataset.testset);
            double[] pre = eval.evaluateModel(randomForest,myDataset.testset);
            for(int i=0;i<pre.length;i++){
                pre[i] = Math.round(pre[i]);
            }
            for(double num:pre){
                System.out.println(num);
            }

            int labelAttributeIndex = myDataset.testset.numAttributes()-1;
            Attribute labelAttribute = myDataset.testset.attribute(labelAttributeIndex);

            for(int i=0;i<myDataset.testset.numInstances();i++){
                System.out.println(myDataset.testset.get(i));
            }
            // 输出准确率
//            System.out.println(eval.toSummaryString("title",true));
//            System.out.println(eval.pctCorrect());
//            System.out.println(eval.pctIncorrect());


        }catch (Exception e){
            e.printStackTrace();
        }
    }


    public static void main(String[] args) {
        MyDataset myDataset = new MyDataset("data\\flower_labels.csv");
        myDataset.dataset_Partitioning(0.9);
        System.out.println(myDataset.testset);
        RandomForest(myDataset,100,0,1,0);
//        MyDataset myDataset = new MyDataset("data\flower.csv");
//        K_means(myDataset.dataset,3);
    }
}
