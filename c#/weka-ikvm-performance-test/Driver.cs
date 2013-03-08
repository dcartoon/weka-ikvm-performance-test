using System;
using System.Collections.Generic;
using ikvm.extensions;
using java.io;
using Console = System.Console;
using IOException = System.IO.IOException;

namespace weka_ikvm_performance_test
{
	/// <summary>
	/// Code to run/evaluate the Weka J48 classifier. 
	/// 
	/// This is largely copied from: http://weka.wikispaces.com/IKVM+with+Weka+tutorial
	/// </summary>
	public class Driver
	{
		private const int NumIterations = 10;
		private const int PercentSplit = 90;
		private const string TrainingTimeOutput = "clr_training_times.csv";
		private const string TestTimeOutput = "clr_test_times.csv";
		private const bool UseRandomForest = false;
		private const int NumTrees = 20;

		public static void Main(String[] args)
		{
			Console.Out.WriteLine("Starting...");

			if(UseRandomForest)
				Console.Out.WriteLine("Using RandomForest with NumTrees=" + NumTrees);
			else
				Console.Out.WriteLine("Using J48 Decision Tree");
			
			weka.core.Instances insts;

			try
			{
				insts = new weka.core.Instances(new java.io.FileReader("..\\..\\..\\..\\data\\adult.data.converted.arff"));
			}
			catch (Exception ex)
			{
				Console.Out.Write(ex.Message);
				return;
			}

			Driver.ClassifyTest(insts);
		}

		public static void ClassifyTest(weka.core.Instances insts)
		{
			try
			{
				List<long> trainingTimes = new List<long>();
				List<long> classificationTimes = new List<long>();

				insts.setClassIndex(insts.numAttributes() - 1);
				int trainSize = insts.numInstances() * PercentSplit/100;
				int testSize = insts.numInstances() - trainSize;

				weka.classifiers.Classifier cl = null;

				Console.Out.WriteLine("Performing " + PercentSplit + "% split evaluation.");

				for (int i = 0; i < NumIterations; i++)
				{
					long startTime = CurrentTimeMillis();

					cl = GetClassifier();
					
					//randomize the order of the instances in the dataset.
					weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
					myRandom.setInputFormat(insts);
					insts = weka.filters.Filter.useFilter(insts, myRandom);

					weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

					cl.buildClassifier(train);
					long endTime = CurrentTimeMillis();
					trainingTimes.Add(endTime - startTime);
				}

				Console.Out.WriteLine("Average training time: " + Driver.ComputeAverage(trainingTimes, true) + "ms");

				for (int i = 0; i < NumIterations; i++)
				{
					long startTime = CurrentTimeMillis();

					int numCorrect = 0;
					for (int j = trainSize; j < insts.numInstances(); j++)
					{
						weka.core.Instance currentInst = insts.instance(j);
						double predictedClass = cl.classifyInstance(currentInst);
						if (predictedClass == insts.instance(j).classValue())
							numCorrect++;
					}
					if (i == 0)
					{
						Console.Out.WriteLine(numCorrect + " out of " + testSize + " correct (" +
								(double) ((double) numCorrect/(double) testSize*100.0) + "%)");
					}

					long endTime = CurrentTimeMillis();
					classificationTimes.Add(endTime - startTime);
				}

				Console.Out.WriteLine("Average classification time: " + Driver.ComputeAverage(classificationTimes, true) +
						"ms");

				Driver.DumpToFile(trainingTimes, Driver.TrainingTimeOutput);
				Driver.DumpToFile(classificationTimes, Driver.TestTimeOutput);
			}
			catch (Exception ex)
			{
				Console.Out.WriteLine(ex.Message);
			}
		}

		private static weka.classifiers.Classifier GetClassifier()
		{
			if (UseRandomForest)
			{
				weka.classifiers.trees.RandomForest forest = new weka.classifiers.trees.RandomForest();
				forest.setNumTrees(NumTrees);
				return forest;
			}

			return new weka.classifiers.trees.J48();
		}
		
		private static long CurrentTimeMillis()
		{
			return DateTime.Now.Ticks/10000;
		}

		private static void DumpToFile(List<long> data, String fileName)
		{
			try
			{
				BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
				foreach(long value in data)
					writer.write(value + "\n");

				writer.close();
			}
			catch (IOException e)
			{
				Console.Out.WriteLine("Failed to write to file: " + fileName);
				e.printStackTrace();
			}
		}

		/// <summary>
		/// Compute the average from a set of data.  Supports skipping the first measurement of code execution tends to be 
		/// off due to the JIT
		/// </summary>
		/// <param name="data"></param>
		/// <param name="skipFirst"></param>
		/// <returns></returns>
		private static float ComputeAverage(List<long> data, bool skipFirst)
		{
			int startIndex = skipFirst ? 1 : 0;
			long result = 0;
			long numResults = 0;
			
			for (int i = startIndex; i < data.Count; i++)
			{
				result += data[i];
				numResults++;
			}

			if (numResults == 0)
			{
				return float.NaN;
			}

			return (float) result/numResults;
		}

	}
}