import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.json.JSONObject;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Xgboost {
    private static DMatrix trainMat = null;
    private static DMatrix testMat = null;
    public static JSONObject xgboost(double eta, int depth, int nEpoch)throws Exception{
            trainMat = new DMatrix("data\\train.txt");
            testMat = new DMatrix("data\\test.txt");

        Map<String,Object> params = new HashMap<String,Object>(){
            {
                put("eta",eta);//为了防止过拟合，更新过程中用到的收缩的步长
                put("max_depth",depth);//树的最大深度
                put("num_class",3);
                put("objective","multi:softmax");
                put("eval_metric","mlogloss");
            }
        };

        Map<String,DMatrix> watches = new HashMap<String,DMatrix>(){
            {
                put("train",trainMat);
//                put("test",testMat);
            }
        };

        Booster booster = XGBoost.train(trainMat,params,nEpoch,watches,null,null);

        float[][] testPred = booster.predict(testMat);
        int[] org_labels = get_orglabel(testMat);
        int[] pred_labels = get_predlabel(testPred);
        Matrix matrix = new Matrix(org_labels,pred_labels);
        double[] all_per  = matrix.get_all();
        for(double per:all_per){
                System.out.println(per);
            }

        Map<String,Double> per_model = new HashMap<>();
        per_model.put("pecision",all_per[0]);
        per_model.put("recall",all_per[1]);
        per_model.put("f1score",all_per[2]);

        booster.saveModel("Intermediate_steps_file\\xgboost_model");

        JSONObject jsonObject = new JSONObject();
        jsonObject.put("performance",per_model);
        return jsonObject;
    }

    public static int[] get_orglabel(DMatrix dMatrix)throws Exception{
            float[] list = dMatrix.getLabel();
            int[] org_label = new int[list.length];
            for(int i=0;i<list.length;i++) {
                org_label[i] = (int) list[i];
//                System.out.println(org_label[i]);
            }
            return org_label;
    }

    public static int[] get_predlabel(float[][] predictions){
        int[] pred_label = new int[predictions.length];
        for(int i=0;i<predictions.length;i++){
            int predictedLabel = Math.round(predictions[i][0]);
            pred_label[i] = predictedLabel;
        }
        return pred_label;
    }


    public static void main(String[] args) throws Exception{
        xgboost(0.1,6,50);//全由前端提供
    }

}