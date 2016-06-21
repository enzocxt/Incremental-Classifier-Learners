/**
 @author Tao Chen

 This class is the class for data process.
 */
import java.io.*;
import java.util.*;
import java.lang.Error;

public class Data{

    private int[][] testData;
    private int[][] trainData;
    //private int[] numValues;
    private int classPosition;
    private int currentIndex;
    private int fvLength;

    /**
        This constructor reads in both the training and testing data.

        @param trainFile the name of the file containing the training data
        @param testFile the name of the file containing the test data
        @param sep is the seperator between the attributes in the
        feature vectors

    */
    public Data(String trainFile, String testFile, String sep) {
        try{
            trainData = readDataFileInt(trainFile, sep);
        }catch(java.io.IOException e){
            System.err.println("Unable training dataset file :" + e.getMessage());
        }

        try{
            testData = readDataFileInt(testFile, sep);
        }
        catch(java.io.IOException e){
            System.err.println("Unable training dataset file :" + e.getMessage());
        }


        currentIndex = 0;
        classPosition = trainData[0].length-1;
        fvLength = trainData[0].length;
    }

    /**
       @return true if there are still training examples to process
     */
    public boolean hasMoreTrainData(){
        if (currentIndex < trainData.length){
            return true;
        }
        else{
            return false;
        }
    }

    /**
       This method returns the next num training examples. If there
       are fewer than num examples left, it returns those that are
       left. If there are no examples left, "null" is returned.

       @param num the number of requested training examples
       @return the next num training examples. 
     */
    public int[][] getTrainExamples(int num){
        if (currentIndex >= trainData.length)
            return null;

        int max = Math.min(num, (trainData.length - currentIndex));
        int[][] examples = new int[max][fvLength];
        for(int ix = 0; ix < max; ix++){
            System.arraycopy(trainData[ix+currentIndex], 0, examples[ix], 0, fvLength);
        }

        currentIndex += max;
        return examples;
    }

    /**
       @return the test data
     */
    public int[][] getTestData(){
	    return testData;
    }

    
    /**
       @return the string representation of any 2D integer vector
    */
    public static String dataToString(int [][] data){
        String output = new String();
        for (int[] t : data){
            for(int v : t)
            output += v+", ";
            output += "\n";
        }
        return output;
    }

    /** 
	@return the string representation of the test set and the training set 
    */
    public String toString(){
        String output = new String();
        output += "Training set: (class position: "+(classPosition+1)+") \n";
        output += dataToString(trainData);
        output += "\n";
        output += "Test set: \n";
        output += dataToString(testData);
        return output;
    }

    /**
       This method reads in a dataset. It is assumed that the data is
       stored as integers or only contains discrete attributes. It
       stores the data as a 2-D integer array

       @param fileName the name of the file containing data set
       @param sep is the seperator between the attributes in the
       feature vectors
 
    */
    private int[][] readDataFileInt(String fileName, String sep) throws java.io.IOException {
        RandomAccessFile infile = new RandomAccessFile(fileName, "r");
        String[] split;
        String line = infile.readLine();
        int count = 1;
        LinkedList<int[]> exs = new LinkedList<int[]>();
        while(line != null){
            split = line.split(sep);
            int[] tmp = new int[split.length];
            for(int ix = 0; ix < split.length; ix++){
            tmp[ix] = Integer.parseInt(split[ix]);
            }//end of for loop
            line = infile.readLine();
            exs.add(tmp);
            count++;
        }
        return convertListInts(exs);
    }//end of method 

    /**
       Converts the initial list of feature vectors into an 2-D array

       @param list is a list of integer arrays
       @return a 2-D integer array
    */
    private int[][] convertListInts(LinkedList<int[]> list){
        int len = list.size();
        int[][] examples = new int[len][];
        for(int ix = 0; ix < len; ix++){
            examples[ix] = list.removeFirst();
        }
        return examples;
    }//end method convertListInts

    
    public static void main(String[] args){
        if (args.length < 3){
            throw new Error("Expected 3 arguments, got "+args.length+".");
        }

        Data d = new Data(args[0], args[1], args[2]);
        System.out.println(d.toString());
    }
}

