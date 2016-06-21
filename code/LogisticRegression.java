/**
   @author Tao Chen

   This class is a stub for incrementally building a Logistic Regression model.
 */
import java.io.*;
import java.util.*;

public class LogisticRegression{

    private int numFeatures;
    private int classPosition;
    private int examplesProcessed;
    private double learningRate;

    private double[] weights;
    private double Epsilon; // error bound
    private double lamda; // penalty parameter of weights
    private int maxIteration; //max number of interation

    /**
       LogisticRegression constructor.

       @param numFeatures is the number of features. 
       @param classPosition position of the label in the feature vector.
       @param learningRate is the learning rate
     */
    public LogisticRegression(int numFeatures, int classPosition, double learningRate){
        this.numFeatures = numFeatures;
        this.classPosition = classPosition;
        this.learningRate = learningRate;
        examplesProcessed = 0;

        this.weights = new double[this.numFeatures];
        for (int i=0; i<this.numFeatures; i++) {
            this.weights[i] = 1.0;
        }
        this.Epsilon = 0.005;
        this.lamda = 0.2;
        this.maxIteration = 10000;
    }

    /**
     * Transform example to X vector with X_0=1
     * @param example
     * @return x vector
     */
    public int[] ex2x(int[] example) {
        int[] x = new int[this.numFeatures];
        x[0] = 1;
        for (int i=1; i<this.numFeatures; i++) {
            x[i] = example[i-1];
        }
        return x;
    }

    /**
     * Linear transformation of x by weights vector
     * @param weights
     * @param x
     * @return sum = weights * x
     */
    public double linear_trans(double[] weights, int[] x) {
        double sum = 0.0;
        for (int i=0; i<x.length; i++) {
            sum += weights[i] * x[i];
        }
        return sum;
    }

    /**
     * sigmoid function
     */
    public double sigmoid(double Zeta){
        return 1.0/(1.0 + Math.exp(-1.0 * Zeta));
    }

    /**
     * J(theta), and for testing the choosing of parameters (learning rate, epsilon, lamda)
     * @param examples
     * @return
     */

    public double Jtheta(int[][] examples) {
        double obj = 0.0;
        for (int i=0; i<examples.length; i++) {
            int y = examples[i][this.classPosition];
            if (makePrediction(examples[i]) == 0) {
                obj += 0.0;
            } else if (makePrediction(examples[i]) == 1) {
                obj += 0.0;
            } else {
                obj += -1.0 * ((double) y * Math.log(makePrediction(examples[i]))
                        + (double) (1-y) * Math.log(1.0-makePrediction(examples[i])));
            }
        }
        return obj;
    }

    /**

       This method updates the parameters of your model using new training examples.

       Uses the training data to update the current parameters of the model. 

       @param examples is a set of training examples
     */
    public void updateParameters(int[][] examples){
        examplesProcessed += examples.length;

        double error = 0.0;
        double jtheta0 = 1000000.0;
        for (int it=0; it<this.maxIteration; it++) {
            for (int i=0; i<examples.length; i++) {
                int[] x = ex2x(examples[i]);
                double predicted = makePrediction(x);
                int label = examples[i][this.classPosition];

    			/* decrease learning rate when iteration and examples increase */
                double alpha = 0.0;
                alpha = 1.0 / (1.0 + (double) it + (double) (i+examplesProcessed)) + this.learningRate;
                double temp = 1.0 / (1.0 + (double) it + (double) (i+examplesProcessed)) +
                        1.0 / ((double) examplesProcessed);
                alpha = Math.min(this.learningRate, temp);

                //weights[0] = weights[0] - this.learningRate * (predicted - (double) label);
                weights[0] = weights[0] - alpha * (predicted - (double) label);
                for (int j=1; j<weights.length; j++) {
                    //weights[j] = weights[j] - this.learningRate * x[j] * (predicted - label);
                    //weights[j] = weights[j] - alpha * x[j] * (predicted - (double) label);
                    //double tmp = alpha * x[j] * (predicted - (double) label);
                    weights[j] = weights[j] - alpha * x[j] * (predicted - (double) label)
                            - alpha * this.lamda * weights[j];
                    //weights[j] = weights[j] - tmp
                    //		- (((double) examplesProcessed) * alpha * this.lamda * 0.5) * alpha * weights[j];
                }
            }
            double jtheta = Jtheta(examples);// new Jtheta
            error = jtheta0 - jtheta; // previous - new, error should > 0
            jtheta0 = jtheta;

            // if J(theta) change < epsilon
            if (jtheta != 0.0 && (Math.abs(error) < (Epsilon) || it==(this.maxIteration-1))) {
                break;
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

        //double Zeta = 0.0;
        //Zeta += linear_trans(weights, ex2x(example));

        //return sigmoid(Zeta);
        return sigmoid(linear_trans(weights, ex2x(example)));
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
	try{
	    RandomAccessFile out = new RandomAccessFile(file + examplesProcessed + ".probs", "rw");	
	    for(int[] testInstance : data){
		out.writeBytes(makePrediction(testInstance) + "\t" +
			       testInstance[classPosition] + "\n");
	    }
	}//end try block
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
	if (args.length < 5){
	    System.err.println("Usage: java LogisticRegression <learningRate> <training set> <testset> <output file> <increment>");
	    throw new Error("Expected 4 arguments, got "+args.length+"."); 
	}
	Data data = new Data(args[1], args[2], ",");
	int[][] testData = data.getTestData();
	LogisticRegression nb = new LogisticRegression(testData[0].length, testData[0].length-1, Double.parseDouble(args[0]));
	int streamSize = Integer.parseInt(args[4]);
	System.out.println("Start training");
	while (data.hasMoreTrainData()){
	    /* Process a new training example and measure the accuracy
	     * on the whole test set.*/
	    int[][] nextExamples = data.getTrainExamples(streamSize);  
	    nb.updateParameters(nextExamples);
	    System.out.println(nb.examplesProcessed + 
			       " example(s) processed, accuracy on the test set: " + 
			       nb.computeAccuracy(testData, 0.5)*100 + "%.");
	    nb.writeAccuracyToFile(args[3], testData, 0.5);
	    nb.writeAllPredictionsToFile(args[3], testData);
	    streamSize *= Integer.parseInt(args[4]);
	}
	
    }

}
