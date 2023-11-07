import java.lang.reflect.Array;
import java.util.Arrays;

public class Matrix {
    int[][] confusionMatrix;

    public Matrix(int[] org,int[] pre){
        int numClassess = Arrays.stream(org).max().getAsInt()+1;
        confusionMatrix = new int[numClassess][numClassess];

        for(int i=0;i<org.length;i++){
            int trueClass = org[i];
            int predClass = pre[i];
            confusionMatrix[trueClass][predClass]++;
        }

        for (int[] row : confusionMatrix) {
            System.out.println(Arrays.toString(row));
        }
    }

    public double[] get_all(){
        int numClasses = this.confusionMatrix.length;

        double[] precision = new double[numClasses];
        double[] recall = new double[numClasses];
        double[] f1score = new double[numClasses];

        for(int i=0;i<numClasses;i++){
            int tp = confusionMatrix[i][i];
            int fp = 0;
            int fn = 0;

            for(int j=0;j<numClasses;j++){
                if(j!=i){
                    fp+=confusionMatrix[j][i];
                    fn+=confusionMatrix[i][j];
                }
            }

            precision[i] = (double)tp/(tp+fp);
            recall[i] = (double)tp/(tp+fn);
            f1score[i] = 2*(precision[i]*recall[i])/(precision[i]+recall[i]);
        }
        double averagePrecision = Arrays.stream(precision).average().orElse(0);
        double averageRecall = Arrays.stream(recall).average().orElse(0);
        double averagef1score = Arrays.stream(f1score).average().orElse(0);

        double[] all_pre = new double[3];
        all_pre[0] = averagePrecision;
        all_pre[1] = averageRecall;
        all_pre[2] = averagef1score;

        return all_pre;
    }





}
