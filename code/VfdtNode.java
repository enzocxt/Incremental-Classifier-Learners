/**
   @author Tao Chen

   This class is a stub for VFDT.
 */
import java.io.*;
import java.util.*;
public class VfdtNode{

    private VfdtNode left; /* left child (null if node is a leaf) */
    private VfdtNode right; /* right child (null if node is a leaf) */

    private int splitFeatureId; /* splitting feature */

    protected int[][][] nijk; /* instance counts (see paper) */

    public int[] X; /* attribute set of this node; if the value is 1, then the attribute is chosen */
    public boolean pure; /* if the examples in this node are all of the same class, pure = true. */
    public int N; /* the number of examples seen at l */
    public int classAttr; // classAttr = most frequent class
    public int pos;
    public int neg;

    /**
     Create and initialize a leaf node.
     */
    public VfdtNode(int numFeatures){
        left = null;
        right = null;

        splitFeatureId = numFeatures-1; /* no split */
        pure = true;
        N = 0;
        classAttr = 2;
        pos = 0;
        neg = 0;
	    /* initialize attribute set; all attributes are not chosen */
        X = new int[numFeatures];
        for (int i=0; i<numFeatures; i++) {
            X[i] = 1;
        }// end of for loop
	    /* initialize nijk[][][] */
        nijk = new int[numFeatures][2][2];
        for (int i=0; i<numFeatures; i++)
            for (int j=0; j<2; j++)
                for (int k=0; k<2; k++)
                    nijk[i][j][k] = 0;
	
    }

    /** 
     Turn a leaf node into a internal node.

     @param testFeature is the feature to test on this node.
     @param left is the left child (testFeature = 0).
     @param right is the right child (testFeature = 1).
     */
    public void addChildren(int testFeature, VfdtNode left, VfdtNode right){

        this.splitFeatureId = testFeature;
        this.left = left;
        this.right = right;
        // For each child, let Xchild = X - {testFeature}
        for (int i=0; i<X.length; i++) {
            if (this.X[i] == 0) {
                this.left.X[i] = this.X[i];
                this.right.X[i] = this.X[i];
            }
        }
        this.left.X[testFeature] = 0;
        this.right.X[testFeature] = 0;

        // Let G(Xnull) be the G obtained by predicting the most frequent class at child
        // have no idea how to implement this ???
	
    }

    /** 
	 Returns the leaf node corresponding to the test example.

	 @param example is the test example to sort.
     */
    public VfdtNode sortExample(int[] example){

        //VfdtNode leaf = null; // change this

	    /* FILL IN HERE */
        if (left == null) {
            return this;
        }
        if (example[splitFeatureId] == 0) {
            if (left.left == null) {
                return left;
            } else {
                return left.sortExample(example);
            }
        } else {
            if (right.left == null) {
                return right;
            } else {
                return right.sortExample(example);
            }
        }

	    //return leaf;
    }

    /**
     * Update statistics: nijk, pure, classAttr
     * @param example
     * @param classPosition
     */
    public void updateStat(int[] example, int classPosition) {
        this.N += 1;
        // Update nijk
        for (int i=0; i<example.length-1; i++)
            if (this.X[i] == 1)
                this.nijk[i][example[i]][example[classPosition]]++;
        // Update pos, neg
        if (example[classPosition] == 0) {
            neg++;
        } else {
            pos++;
        }
        // Update classAttr and pure
        if (this.N > 1 && this.pure == true && classAttr != example[classPosition]) {
            this.pure = false;
        }
        if (pos > neg) {
            this.classAttr = 1;
        } else {
            this.classAttr = 0;
        }
    }

    /**
       Split evaluation method (function G in the paper)
       
       Compute a splitting score for the feature featureId.
       For now, we'll use information gain, but this may be changed. 
       
       @param featureId is the feature to be considered. 
    */
    public double splitEval(int featureId){
	    return informationGain(featureId, nijk);
    }

    /**
       Compute the information gain of a feature for this leaf node.

       @param featureId is the feature to be considered. 
       @param nijk are the instance counts.
    */ 
    public static double informationGain(int featureId, int[][][] nijk){
	    double ig = 0;

        double entropyS = 0.0;
        int pos = 0;
        int neg = 0;
        double posPr = 0;
        double negPr = 0;
        for (int j=0; j<2; j++) {
            pos += nijk[featureId][j][1];
            neg += nijk[featureId][j][0];
        }
        posPr = (double) pos / (double) (pos + neg);
        negPr = (double) neg / (double) (pos + neg);

        double entropyNew = 0.0;
        double[] entropySv = new double[2];
        for (int j=0; j<2; j++) {
            int numSv = 0;
            numSv = nijk[featureId][j][0] + nijk[featureId][j][1];
            if (numSv == 0) {
                entropySv[j] = 0.0;
                continue;
            }
            if (nijk[featureId][j][1] == 0) {
                entropySv[j] = -1.0 * ((double) nijk[featureId][j][0]/(double) numSv) * (Math.log((double) nijk[featureId][j][0]/(double) numSv)/Math.log(2));
            } else if (nijk[featureId][j][0] == 0) {
                entropySv[j] = -1.0 * ((double) nijk[featureId][j][1]/(double) numSv) * (Math.log((double) nijk[featureId][j][1]/(double) numSv)/Math.log(2));
            } else {
                entropySv[j] = -1.0 * ((double) nijk[featureId][j][1]/(double) numSv) * (Math.log((double) nijk[featureId][j][1]/(double) numSv)/Math.log(2))
                        - ((double) nijk[featureId][j][0]/(double) numSv) * (Math.log((double) nijk[featureId][j][0]/(double) numSv)/Math.log(2));
            }
            entropyNew += entropySv[j];
        }
        if (posPr == 0 || negPr == 0) {
            entropyS = 0.0;
        } else {
            entropyS = -1.0 * posPr * (Math.log(posPr)/Math.log(2)) - negPr * (Math.log(negPr)/Math.log(2));
        }
        /*if (this.pure == true) {
            entropyS = 0.0;
        } else {
            entropyS = -1.0 * posPr * (Math.log(posPr)/Math.log(2)) - negPr * (Math.log(negPr)/Math.log(2));
        }*/
        ig = entropyS - entropyNew;
	
	    return ig;
    }


    public void printTree(){
	    printTree("");
    }

    private void printTree(String indent){
        if (left==null || right==null){
            System.out.println(indent+"Leaf");
        }
        else {
            System.out.println(indent+splitFeatureId+"=0:");
            left.printTree(indent+"| ");
            System.out.println(indent+splitFeatureId+"=1:");
            right.printTree(indent+"| ");
        }
    }

}
