import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.IOException;

class MyDataset {
    Instances dataset;
    Instances trainset;
    Instances testset;

    public void getDatasetfromCSV(String csvpath) {
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(csvpath));
            Instances datas = loader.getDataSet();
            this.dataset = datas;
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void dataset_Partitioning(double trainRatio){
        try{
//            double percent = train_len/(train_len+test_len);
            int trainSize = (int)(this.dataset.numInstances()*trainRatio);
            int testSize = this.dataset.numInstances()-trainSize;

            this.trainset = new Instances(this.dataset,0,trainSize);
            this.testset = new Instances(this.dataset,trainSize,testSize);

            this.trainset.setClassIndex(this.trainset.numAttributes()-1);
            this.testset.setClassIndex(this.testset.numAttributes()-1);
//            System.out.println(this.testset.numAttributes()-1);
        }catch (Exception e){
            e.printStackTrace();
        }

    }

//    public Instances[] dataset_Partitioning(String dataFilePath, double trainRatio) throws IOException{
//        int trainSize = (int)(this.dataset.numInstances()*trainRatio);
//        int testSize = this.dataset.numInstances()-trainSize;
//
//        RemovePercentage removePercentage = new RemovePercentage();
//        removePercentage.setInputFormat(this.dataset);
//    }

    //返回最后一列的序号数
    public int getLabelIndex(){
        int labelindex = 0;
        try{
            labelindex = this.dataset.numAttributes()-1;
        }catch(Exception e){
            e.printStackTrace();
        }
        return labelindex;
    }


    //返回标签种类数
    public int getNuminput(){
        int Numinput = 0;
        try{
            int labelIndex = this.dataset.numAttributes()-1;
            AttributeStats labelStats = this.dataset.attributeStats(labelIndex);
            Numinput = labelStats.distinctCount;
        }catch (Exception e){
            e.printStackTrace();
        }
        return Numinput;
    }

    public MyDataset(String path){
        if(path.charAt(path.length()-1)=='v'){
            getDatasetfromCSV(path);
        }
    }

    public static void main(String[] args) {
        MyDataset myDataset = new MyDataset("data\\flower_labels.csv");
        System.out.println(myDataset.getNuminput());
//        myDataset.dataset_Partitioning(9,1);
    }

}