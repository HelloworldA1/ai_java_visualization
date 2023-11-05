import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.util.HashMap;
import java.util.Map;

public class Xgboost {
    private static DMatrix trainMat = null;
    private static DMatrix testMat = null;
    public static void xgboost(double eta,int depth,int nEpoch){
        try{
            trainMat = new DMatrix("data\\train.txt");
        } catch (XGBoostError xgBoostError) {
            xgBoostError.printStackTrace();
        }
        try{
            testMat = new DMatrix("data\\test.txt");
        }catch (XGBoostError xgBoostError){
            xgBoostError.printStackTrace();
        }

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
                put("test",testMat);
            }
        };

        try{
            Booster booster = XGBoost.train(trainMat,params,nEpoch,watches,null,null);
            booster.saveModel("Intermediate_steps_file\\xgboost_model");
        }catch (XGBoostError xgBoostError){
            xgBoostError.printStackTrace();
        }

    }

    public static void main(String[] args) {
        xgboost(0.1,6,50);
    }

}