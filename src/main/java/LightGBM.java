import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMDataset;
import io.github.metarank.lightgbm4j.LGBMException;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

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

    public static void lightgbm(String path,int nEpoch) throws Exception {
        LGBMDataset dataset = data(path);
        LGBMBooster booster = LGBMBooster.create(dataset,"objective-binary label=name:Classification");
        for(int i=0;i<10;i++){
            booster.updateOneIter();
            double[] eval =booster.getEval(0);
            System.out.println(eval[0]);
            assertTrue(eval[0]>0);
        }
        String[] names = booster.getFeatureNames();
        double[] weights = booster.featureImportance(0, LGBMBooster.FeatureImportanceType.GAIN);
        assertTrue(names.length > 0);
        assertTrue(weights.length > 0);
        booster.close();
        dataset.close();
    }

    public static void main(String[] args) throws Exception{
        lightgbm("data\\\\flower_labels.csv",50);
    }
}