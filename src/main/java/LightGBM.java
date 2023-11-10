import com.microsoft.ml.lightgbm.PredictionType;
import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMDataset;
import io.github.metarank.lightgbm4j.LGBMException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertTrue;

public class LightGBM {

    public static String[] get_feature_name(String csvpath)throws Exception{
        String[] feature_name;
        BufferedReader reader = new BufferedReader(new FileReader(csvpath));
        String line = reader.readLine();
        boolean ishead = true;
        feature_name = line.split(",");
        String[] finallist = new String[feature_name.length-1];
        for(int i=0;i<feature_name.length-1;i++){
            finallist[i] = feature_name[i];
        }
        return finallist;
    }

    public static double[] get_value(String csvpath) throws IOException {
        ArrayList<Double> all_value= new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(csvpath));
        String line;
        boolean ishead = true;
        while ((line = reader.readLine())!=null){
//            System.out.println(line);
            if(ishead){
                ishead = false;
                continue;
            }else{
                String[] parts = line.split(",");
                for(int i=0;i<parts.length-1;i++){
                    double value = Double.parseDouble(parts[i]);
                    all_value.add(value);
                }
            }
        }
        double[] values = new double[all_value.size()];
        for(int i=0;i<all_value.size();i++){
            values[i] = all_value.get(i);
        }
        return values;
    }

    public static float[] get_labels(String csvpath)throws IOException {
        ArrayList<Float> all_value = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(csvpath));
        String line;
        boolean ishead = true;
        while ((line = reader.readLine())!=null){
//            System.out.println(line);
            if(ishead){
                ishead = false;
                continue;
            }else{
                String[] parts = line.split(",");
                float label = Float.parseFloat(parts[parts.length-1]);
                all_value.add(label);
            }
        }
        float[] labels = new float[all_value.size()];
        for(int i=0;i<all_value.size();i++){
            labels[i]=all_value.get(i);
        }
        return labels;
    }


    //    "data\\flower_labels.csv"
    public static LGBMDataset data(String path) throws Exception {
        String[] colums = get_feature_name(path);
        double[] values = get_value(path);
        float[] label = get_labels(path);
        LGBMDataset dataset = LGBMDataset.createFromMat(values,label.length,colums.length,true,"",null);
        dataset.setFeatureNames(colums);
        dataset.setField("label",label);
        return dataset;
    }

//    public static LGBMDataset testdata(String path,LGBMDataset traindata) throws Exception {
//        String[] colums = get_feature_name(path);
//        double[] values = get_value(path);
//        float[] label = get_labels(path);
//        LGBMDataset dataset = LGBMDataset.createFromMat(values,label.length,colums.length,true,"",traindata);
//        dataset.setFeatureNames(colums);
//        dataset.setField("label",label);
//        return dataset;
//    }

    public static JSONObject lightgbm(int nEpoch) throws Exception {
        String trainpath = "data\\flower_labels_lightgbm.csv";
        String testpath = "data\\flower_labels_lightgbm_test.csv";
        LGBMDataset traindataset = data(trainpath);
        LGBMDataset testdataset = data(testpath);
        LGBMBooster booster = LGBMBooster.create(traindataset,"objective=multiclass num_class=3");
//        booster.addValidData(testdataset);
        Map<Integer,Double> loss = new HashMap<>();
        for(int i=0;i<nEpoch;i++){
            booster.updateOneIter();
            double[] eval =booster.getEval(0);
            System.out.println("-------start--------");
            System.out.println(eval[0]);
            loss.put(i,eval[0]);
            System.out.println("-------end--------");
            assertTrue(eval[0]>0);
        }
        int[] org_label = getorg_label();
        int[] pred_label = getpred_label(testdataset,booster);
        Matrix matrix = new Matrix(org_label,pred_label);
        double[] all_per =  matrix.get_all();
        for(double item:all_per){
            System.out.println(item);
        }
        Map<String,Double> per_model = new HashMap<>();
        per_model.put("pecision",all_per[0]);
        per_model.put("recall",all_per[1]);
        per_model.put("f1score",all_per[2]);

        String[] names = booster.getFeatureNames();
        double[] weights = booster.featureImportance(0, LGBMBooster.FeatureImportanceType.GAIN);
        assertTrue(names.length > 0);
        assertTrue(weights.length > 0);
        booster.close();
        traindataset.close();
        testdataset.close();

        JSONObject jsonObject = new JSONObject();
        jsonObject.put("loss",loss);
        jsonObject.put("performance",per_model);
//        System.out.println(jsonObject.toString());
        return jsonObject;
    }

    public static int[] getorg_label() throws LGBMException, IOException {
        float[] labels = get_labels("data\\flower_labels_lightgbm_test.csv");
        int[] org_labels = new int[labels.length];
        for(int i=0;i<labels.length;i++){
            org_labels[i] = (int)labels[i];
        }
        return org_labels;
    }

    public static double[] getpred_rawscore(LGBMDataset testset,LGBMBooster booster) throws LGBMException, IOException {
        double[] testpred = new double[testset.getNumData()];
        double[] feature = get_value("data\\flower_labels_lightgbm_test.csv");
        float[] features = new float[feature.length];
        for(int i=0;i<feature.length;i++){
            features[i] = (float)feature[i];
        }
        testpred = booster.predictForMat(features,12,4,true, PredictionType.C_API_PREDICT_RAW_SCORE);
        return testpred;
    }

    public static int[] getpred_label(LGBMDataset testset,LGBMBooster booster) throws IOException, LGBMException {
        double[] raw_score = getpred_rawscore(testset,booster);
        int[] labels = new int[12];
        int j=0;
        for(int i = 0; i<raw_score.length; i+=3){
             double l0 = raw_score[i];
             double l1 = raw_score[i+1];
             double l2 = raw_score[i+2];
             int label = find_max(l0,l1,l2);
             labels[j] = label;
             j+=1;
        }
        return labels;
    }

    public static int find_max(double l0,double l1,double l2){
        if (l0 >= l1 && l0 >= l2) {
            return 0;
        } else if (l1 >= l0 && l1 >= l2) {
            return 1;
        } else {
            return 2;
        }
    }

    public static void main(String[] args) throws Exception{
        lightgbm(50);
    }
}