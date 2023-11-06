import org.json.JSONObject;
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
import java.util.HashMap;
import java.util.Map;


public class RandomForest_K_means {
    //k-means
    public static JSONObject K_means(Instances datas,int NUmClusters){
        JSONObject  jsonObject = new JSONObject();
        try{
            SimpleKMeans kMeans = new SimpleKMeans();
            kMeans.setPreserveInstancesOrder(true);
            kMeans.setNumClusters(NUmClusters); // 设置簇的数量
            kMeans.buildClusterer(datas); // 训练模型

            Map<String,String> pre_km = new HashMap<>();

            // 获取聚类中心
            Instances centroids = kMeans.getClusterCentroids();
            for(int i=0;i<NUmClusters;i++) {
                String center = centroids.get(0).toString();
                pre_km.put("center"+Integer.toString(i),center);
            }
            // 获取簇内误差平方和
            double squaredErrors = kMeans.getSquaredError();
            pre_km.put("SE",Double.toString(squaredErrors));

            String cluster_0 = "";
            String cluster_1 = "";
            String cluster_2 = "";

            for (int i = 0; i < datas.numInstances(); i++) {
                int cluster = kMeans.clusterInstance(datas.instance(i));
                if(cluster == 0){
                    cluster_0=cluster_0+Integer.toString(i+1)+",";
                }else if(cluster == 1){
                    cluster_1=cluster_1+Integer.toString(i+1)+",";
                }else{
                    cluster_2 = cluster_2+Integer.toString(i+1)+",";
                }
            }

            pre_km.put("cluster_0",cluster_0);
            pre_km.put("cluster_1",cluster_1);
            pre_km.put("cluster_2",cluster_2);

            jsonObject.put("performance",pre_km);
            System.out.println(pre_km);

        }catch (Exception e){
            e.printStackTrace();
        }
        return jsonObject;
    }


    //randomforest
    public static JSONObject RandomForest(MyDataset myDataset,int NumTrees,int depth,int NumSeed,int NumAttribute)  {
        JSONObject jsonObject = new JSONObject();

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
//                System.out.println(num);
            }

            double[] org = new double[myDataset.testset.numInstances()];

            for(int i=0;i<myDataset.testset.numInstances();i++){
                org[i] = Double.parseDouble(myDataset.testset.get(i).toString().split(",")[4]);
            }

            Map<String,Double> pre_model = new HashMap<>();

//            System.out.println(get_acc(org,pre));

            pre_model.put("ACC",get_acc(org,pre));
//            System.out.println(eval.recall(3));
            // 输出准确率

            System.out.println(eval.toSummaryString("",true));
            for(String item:eval.toSummaryString("",true).split("\n")){
                if(item.contains("Correlation coefficient")){
                    double correlatin = Double.parseDouble(item.split("                  ")[1]);
                    pre_model.put("CC",correlatin);
                }else if(item.contains("Mean absolute error")){
                    double MAE = Double.parseDouble(item.split("                      ")[1]);
                    pre_model.put("MAE",MAE);
                }else if(item.contains("Root mean squared error")){
                    double MSE = Double.parseDouble(item.split("                  ")[1]);
                    pre_model.put("MSE",MSE);
                }else if(item.contains("Relative absolute error")){
                    double RAE = turn_Str_to_Dou(item.split("                  ")[1]);
                    pre_model.put("RAE",RAE);
                }else if(item.contains("Root relative squared error")){
                    double RRSE = turn_Str_to_Dou(item.split("              ")[1]);
                    pre_model.put("RRSE",RRSE);
                    break;
                }
            }
            jsonObject.put("performance",pre_model);

            System.out.println(pre_model);
        }catch (Exception e){
            e.printStackTrace();
        }
        return jsonObject;
    }


    public  static  double get_acc(double[] org,double[] pre){
        int all=org.length;
        int correct = 0;
        for(int i=0;i<all;i++){
            if(org[i]==pre[i]){
                correct+=1;
            }
        }
        double acc = correct/all;
        return acc;
    }

    public static double turn_Str_to_Dou(String precent){
        double number = Double.parseDouble(precent.split(" %")[0])/100;
        return number;
    }


    public static void main(String[] args) {
//        MyDataset myDataset = new MyDataset("data\\flower_labels.csv");
//        myDataset.dataset_Partitioning(0.9);
////        System.out.println(myDataset.testset);
//        RandomForest(myDataset,100,0,1,0);
        MyDataset myDataset = new MyDataset("data\\flower.csv");
        K_means(myDataset.dataset,3);
    }
}
