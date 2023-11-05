import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class RNN {
    public static void csvDealing(String csvpath)throws Exception{
        File input = new File(csvpath);
        RecordReader recordReader = new CSVRecordReader(1,',');
        recordReader.initialize(new FileSplit(input));

        List<String> textData = new ArrayList<>();
        List<Integer> textlabel = new ArrayList<>();

        while(recordReader.hasNext()){
            List<Writable> record = recordReader.next();
            textData.add(record.get(0).toString());
            textlabel.add(record.get(1).toInt());
        }

    }

    public static void main(String[] args) throws Exception {
        csvDealing("data\\train_rnn.csv");
    }
}
