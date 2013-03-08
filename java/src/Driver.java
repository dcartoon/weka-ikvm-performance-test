import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Code to run/evaluate the Weka J48 classifier.
 * 
 * This is largely copied from: http://weka.wikispaces.com/IKVM+with+Weka+tutorial
 * @author dan.cartoon
 *
 */
public class Driver {
	
	private static final int numIterations = 50;
	private static final int percentSplit = 66;
	private static final String trainingTimeOutput = "java_training_times.csv";
	private static final String testTimeOutput = "java_test_times.csv";
	
	public static void main(String[] args) {
		System.out.println("Starting...");
		
		weka.core.Instances insts = null;
		try
	    {
			insts = new weka.core.Instances(new java.io.FileReader("..\\data\\adult.data.converted.arff"));
	    }
		catch (java.lang.Exception ex)
       {
           ex.printStackTrace();
           return;
       }
		
		Driver.classifyTest(insts);
	}
		
    public static void classifyTest(weka.core.Instances insts)
    {
       try
       {
           List<Long> trainingTimes = new ArrayList<Long>();
           List<Long> classificationTimes = new ArrayList<Long>();
                      
           insts.setClassIndex(insts.numAttributes() - 1);
           int trainSize = insts.numInstances() * percentSplit / 100;
           int testSize = insts.numInstances() - trainSize;
           
           weka.classifiers.Classifier cl = null;
           
           System.out.println("Performing " + percentSplit + "% split evaluation.");
           
           for(int i = 0; i < numIterations; i++) {
        	   long startTime = System.currentTimeMillis();
        	   cl = new weka.classifiers.trees.J48();
	       	
	           //randomize the order of the instances in the dataset.
	                       weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
	           myRandom.setInputFormat(insts);
	                       insts = weka.filters.Filter.useFilter(insts, myRandom);
		           
	           weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);
	
	           cl.buildClassifier(train);
	           long endTime = System.currentTimeMillis();
	           trainingTimes.add(endTime - startTime);
           }

           System.out.println("Average training time: " + Driver.computeAverage(trainingTimes, true) + "ms");
           
           for(int i = 0; i < numIterations; i++) {
        	   long startTime = System.currentTimeMillis();
        	   
        	   int numCorrect = 0;
	           for (int j = trainSize; j < insts.numInstances(); j++)
	           {
	               weka.core.Instance currentInst = insts.instance(j);
	               double predictedClass = cl.classifyInstance(currentInst);
	               if (predictedClass == insts.instance(j).classValue())
	                   numCorrect++;
	           }
	           if(i == 0) {
	        	   System.out.println(numCorrect + " out of " + testSize + " correct (" +
	                       (double)((double)numCorrect / (double)testSize * 100.0) + "%)");
	           }
	           
	           long endTime = System.currentTimeMillis();
	           classificationTimes.add(endTime - startTime);
           }
           
           System.out.println("Average classification time: " + Driver.computeAverage(classificationTimes, true) + 
        		   "ms");
           
           Driver.dumpToFile(trainingTimes, Driver.trainingTimeOutput);
           Driver.dumpToFile(classificationTimes, Driver.testTimeOutput);
       }
       catch (java.lang.Exception ex)
       {
           ex.printStackTrace();
       }
   }
   
   private static void dumpToFile(List<Long> data, String fileName) {
	   try {
	        BufferedWriter out = new BufferedWriter(new FileWriter(fileName));
	        
	        for(Long value : data) {
	        	out.write(value + "\n");
	        }
	        out.close();
	    } catch (IOException e) {
	    	System.out.println("Failed to write to file: " + fileName);
	    	e.printStackTrace();
	    }
   }
    
   /**
    * Compute the average from a set of data.  Supports skipping the first measurement of code execution tends to be 
    * off due to the JIT
    * 
    * @param times
    * @param skipFirst
    * @return
    */
   private static float computeAverage(List<Long> data, boolean skipFirst) {
	   int startIndex = skipFirst ? 1 : 0;
	   long result = 0;
	   long numResults = 0;
	   
	   for(int i = startIndex; i < data.size(); i++) {
		   result += data.get(i);
		   numResults++;
	   }
	   
	   if(numResults == 0) {
		   return Float.NaN;
	   }
	   
	   return (float)result/numResults;
   }
	
}
