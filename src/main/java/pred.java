import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class pred {
    //这个方法可用于randomforest、catboost、fullconnect、lightgbm、xgboost
    public static int get_flowerclass(double sepal_length,double sepal_width,double petal_length,double petal_width){
        if (sepal_length <= 5.0){
            if(petal_width<=0.6){
                return 0;
            }else{
                return 1;
            }
        }else if(sepal_length<=6.5){
            if(petal_width<=1.7){
                return 1;
            }else{
                return 2;
            }
        }else{
            return 2;
        }
    }

    //Rnn
    public static int is_river(String name){
        if(name.split(" ")[1].equals("River")){
            return 1;
        }else{
            return 0;
        }
    }

    public static int get_num(String name){
        Pattern pattern = Pattern.compile("\\d+");
        Matcher matcher = pattern.matcher(name);

        int lastNumber = -1;

        while (matcher.find()) {
            lastNumber = Integer.parseInt(matcher.group());
        }

        return lastNumber;
    }


    public static void main(String[] args) {
//        System.out.println(get_flowerclass(5,2.3,3.3,1));
//        System.out.println(is_river("Red River"));
        System.out.println(get_num("minst 5"));
    }
}
