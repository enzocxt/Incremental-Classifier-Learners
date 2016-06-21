/**
 @author Tao Chen

 This class is a stub for incrementally building a Naive Bayes model.

 (c) 2015
 */
import java.io.*;
import java.util.*;
public class NaiveBayes{

    private int numFeatures;
    private int classPosition;
    private int examplesProcessed;

    private int[] classNum;
    private int[][][] num;

    /**
     NaiveBayes constructor.

     @param numFeatures is the number of features.
     @param classPosition position of the label in the feature vector.
     */
    public NaiveBayes(int numFeatures, int classPosition){
        this.numFeatures = numFeatures;
        this.classPosition = classPosition;
        examplesProcessed = 0;

        int i;
        for(i = 0; i < 2; ++i) {
            this.classNum[i] = 0;
        }

        this.num = new int[numFeatures - 1][2][2];

        for(i = 0; i < numFeatures - 1; ++i) {
            for(int j = 0; j < 2; ++j) {
                this.num[i][j][0] = 0;
                this.num[i][j][1] = 0;
            }
        }
    }

    /**

     This method will update the parameters of you model using a new training example.

     Uses the training data to update the current parameters of the model.

     @param examples is a set of training examples
     */
    public void updateParameters(int[][] examples){

        this.examplesProcessed += examples.length;

        for(int n = 0; n < examples.length; ++n) {
            ++this.classNum[examples[n][this.classPosition]];

            for(int i = 0; i < this.numFeatures - 1; ++i) {
                ++this.num[i][examples[n][i]][examples[n][this.classPosition]];
            }
        }
    }

    /**
     Uses the current model to calculate the probability that an
     example belongs to class "1";

     @param example is a test example
     @return the probability that example belongs to class "1"
     */
    public double makePrediction(int[] example){
        double prediction = 0;
        double[] classProb = new double[2];

        for(int Prob = 0; Prob < 2; ++Prob) {
            classProb[Prob] = (double)this.classNum[Prob] / (double)this.examplesProcessed;
        }

        double[][][] var9 = new double[this.numFeatures - 1][2][2];

        int k;
        int i;
        for(int likelihood = 0; likelihood < this.numFeatures - 1; ++likelihood) {
            for(k = 0; k < 2; ++k) {
                for(i = 0; i < 2; ++i) {
                    var9[likelihood][k][i] = (double)this.num[likelihood][k][i] / (double)this.classNum[i];
                }
            }
        }

        double[] var10 = new double[2];
        var10[0] = var10[1] = 1.0D;

        for(k = 0; k < 2; ++k) {
            for(i = 0; i < this.numFeatures - 1; ++i) {
                var10[k] *= var9[i][example[i]][k];
            }
        }

        prediction = classProb[1] * var10[1] / (classProb[1] * var10[1] + classProb[0] * var10[0]);

        return prediction;
    }

    /**
     Use makePrediction() to compute the probability of each example
     in the test set to belongs to class "1".

     The predictions are then written to a file named "file +
     examplexProcessed + .probs".

     The file format has the form of Prob(example[ix] belongs to
     class "1") + tab + true label of example.

     @param file the stem of the output file
     @param data is the test data
     */
    public void writeAllPredictionsToFile(String file, int[][] data){
        /*
        try{
            RandomAccessFile out = new RandomAccessFile(file + examplesProcessed + ".probs", "rw");
            for(int[] testInstance : data){
                out.writeBytes(makePrediction(testInstance) + "\t" +
                        testInstance[classPosition] + "\n");
            }
        }//end try block
        */
        try {
            RandomAccessFile exc = new RandomAccessFile(file + this.examplesProcessed + ".probs", "rw");
            int[][] var7 = data;
            int var6 = data.length;

            for(int var5 = 0; var5 < var6; ++var5) {
                int[] testInstance = var7[var5];
                exc.writeBytes(this.makePrediction(testInstance) + "\t" + testInstance[this.classPosition] + "\n");
            }
        }
        catch(IOException exc){
            System.out.println(exc.toString());
        }
    }

    /**
     Computes accuracy of the current model on the test set.

     A prediction for an instance labeled "1" is considered to be
     correct iff the model predicts the label "1" with probability
     above the probability threshold.

     @param data is the test set.
     @param thres is the probability threshold.
     */
    public double computeAccuracy(int [][]data, double thres){
        double correct = 0;
        for(int ix = 0; ix < data.length; ix++){
            double predict = makePrediction(data[ix]);
            if (((data[ix][classPosition] == 0) && (predict <= thres)) ||
                    ((data[ix][classPosition] == 1) && (predict > thres))){
                correct++;
            }
        }
        return correct / data.length;
    }

    /**
     Computes accuracy of the current model on the test set and
     store the result into a file.

     If the file doesn't exist, a new file named "file + .nb.acc" is
     created.  Each call to this function adds a new line into the
     file. The lines have the form:
     "examplesProcessed <tab> accuracy\n".

     @param file the stem of the output file
     @param data is the test data
     @param thres is the threshold for labeling an example as belonging to class "1"
     */
    public void writeAccuracyToFile(String file, int[][] data, double thres){
        double accuracy = computeAccuracy(data, thres);
        try{
            RandomAccessFile out = new RandomAccessFile(file + ".nb.acc", "rw");
            long fileLength = out.length();
            out.seek(fileLength);
            out.writeBytes(examplesProcessed + "\t" + accuracy + "\n");

        }//end try block
        catch(IOException exc){
            System.out.println(exc.toString());
        }
    }

    /**
     This runs your code to generate the required output for the assignment.
     */
    public static void main(String[] args){
        if (args.length < 4){
            System.err.println("Usage: java NaiveBayes <training set> <testset> <output file> <increment>");
            throw new Error("Expected 4 arguments, got "+args.length+".");
        }
        Data data = new Data(args[0], args[1], ",");
        int[][] testData = data.getTestData();
        NaiveBayes nb = new NaiveBayes(testData[0].length, testData[0].length-1);
        int streamSize = Integer.parseInt(args[3])*data.trainDataSize()/100;
        int processed = 0;

        while (data.hasMoreTrainData()){
	    /* Process a new training example and measure the accuracy
	     * on the whole test set.*/
            int[][] nextExamples = data.getTrainExamples(streamSize);
            nb.updateParameters(nextExamples);
            System.out.println(nb.examplesProcessed +
                    "% processed, accuracy on the test set: " +
                    nb.computeAccuracy(testData, 0.5) * 100 + "%.");
            processed++;
            nb.writeAccuracyToFile(args[2], testData, 0.5, processed);
            //	    nb.writeAllPredictionsToFile(args[2], testData);
        }

    }

}
