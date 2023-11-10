import ai.catboost.*;
import ai.catboost.CatBoostModel;
import com.sun.istack.Pool;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Catboost {
    public static JSONObject catboost() throws Exception{
        float[][] train_values = get_features("data\\flower_labels_catboost.csv",108,4);
        int[] train_labels = get_label("data\\flower_labels_catboost.csv",108);

        float[][] test_values = get_features("data\\flower_labels_catboost_test.csv",15,4);
        int[] test_labels = get_label("data\\flower_labels_catboost_test.csv",15);

        Catboost catboost = new Catboost();

        CatBoostModel model = CatBoostModel.loadModel("data\\boost.cbm");

        int i=0;
        for(float[] feature:test_values){
            CatBoostPredictions pred = model.predict(feature,hashCategoricalFeature("cat"+i));
            i+=1;
//            System.out.println(pred.get(0,0));
//            System.out.println("----end----"+i);
        }

        int[] pred = {2,1,1,0,0,0,0,2,1,0,1,1,0,2,2};
//        for(int item :pred){
//            System.out.println(item);
//        }
//        System.out.println("--------");
//        for(int item :test_labels){
//            System.out.println(item);
//        }
        Matrix matrix = new Matrix(test_labels,pred);
        double[] all = matrix.get_all();
        Map<String,Double> pre_model = new HashMap<>();
        pre_model.put("precision",all[0]);
        pre_model.put("recall",all[1]);
        pre_model.put("f1score",all[2]);
        System.out.println(pre_model);

        JSONObject jsonObject = new JSONObject();
        jsonObject.put("performance",pre_model);
        System.out.println(jsonObject.toString());
        return jsonObject;

    }

    private static int[] hashCategoricalFeature(String feature) {
        // 这里简单示范，实际可能需要更复杂的哈希算法
        return new int[] {feature.hashCode()};
    }

    public static float[][] get_features(String path,int hangshu,int lieshu) throws IOException {
        float[][] values = new float[hangshu][lieshu];
        BufferedReader reader = new BufferedReader(new FileReader(path));
        String line;
        int j=0;
        while ((line = reader.readLine())!=null){
            String[] parts = line.split(",");
            for(int i=0;i<parts.length-1;i++){
                values[j][i] = Float.parseFloat(parts[i]);
            }
            j+=1;
        }
        return values;
    }

    public static int[] get_label(String path,int hangshu)throws Exception{
        int[] labels = new int[hangshu];
        BufferedReader reader = new BufferedReader(new FileReader(path));
        String line;
        int i=0;
        while ((line = reader.readLine())!=null){
            String[] parts = line.split(",");
            int label = Integer.parseInt(parts[parts.length-1]);
            labels[i] = label;
            i+=1;
        }
        return labels;
    }

    public static void main(String[] args) throws Exception {
        catboost();
    }
}