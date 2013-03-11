using System;
using System.Collections.Generic;
using ikvm.extensions;
using java.io;
using java.util.zip;
using weka.classifiers;
using weka.classifiers.xml;
using weka.core;
using weka.core.xml;
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
		private const int NumIterations = 1;
		private const int PercentSplit = 90;
		private const string TrainingTimeOutput = "clr_training_times.csv";
		private const string TestTimeOutput = "clr_test_times.csv";
		private const bool UseRandomForest = true;
		private const int NumTrees = 20;
		private const bool TestClassifierPersistence = true;
		private const string SavedClassifier = "classifier.classifier";

		public static void Main(String[] args)
		{
			Console.Out.WriteLine("Starting...");

			if (UseRandomForest)
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
				int trainSize = insts.numInstances() * PercentSplit / 100;
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

				if (TestClassifierPersistence)
				{
					Console.Out.WriteLine("Persisting and reloading classifier");
					SaveClassifier(cl);
					Classifier persistedClassifier = LoadClassifier();
					cl = persistedClassifier;
				}

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
								(double)((double)numCorrect / (double)testSize * 100.0) + "%)");
					}

					long endTime = CurrentTimeMillis();
					classificationTimes.Add(endTime - startTime);
				}

				double averageTotalClassificationTime = Driver.ComputeAverage(classificationTimes, true);
				double averageIndividualClassificationTime = averageTotalClassificationTime / testSize;
				Console.Out.WriteLine("Average total classification time: " + averageTotalClassificationTime +
						"ms | Average individual classification time: " + averageIndividualClassificationTime);

				Driver.DumpToFile(trainingTimes, Driver.TrainingTimeOutput);
				Driver.DumpToFile(classificationTimes, Driver.TestTimeOutput);

				
			}
			catch (Exception ex)
			{
				Console.Out.WriteLine(ex.Message);
			}
		}

		/// <summary>
		/// Loads a previously persisted classifier.  Mostly taken from Weka source
		/// </summary>
		/// <returns></returns>
		private static weka.classifiers.Classifier LoadClassifier()
		{
			weka.classifiers.Classifier classifier = null;
			ObjectInputStream objectInputStream = null;
			BufferedInputStream xmlInputStream = null;
			const string objectInputFileName = SavedClassifier;
			try {
				if (objectInputFileName.length() != 0) {
					if (objectInputFileName.endsWith(".xml")) {
						// if this is the case then it means that a PMML classifier was
						// successfully loaded earlier in the code
						objectInputStream = null;
						xmlInputStream = null;
					} else {
						InputStream istream = new FileInputStream(objectInputFileName);
						if (objectInputFileName.endsWith(".gz")) 
						{
							istream = new GZIPInputStream(istream);
						}
						// load from KOML?
						if (!(objectInputFileName.endsWith(".koml") && KOML.isPresent())) {
							objectInputStream = new ObjectInputStream(istream);
							xmlInputStream = null;
						} else {
							objectInputStream = null;
							xmlInputStream = new BufferedInputStream(istream);
						}
					}
				}
			} catch (Exception e) {
				throw new Exception("Can't open file " + e.getMessage() + '.');
			}


			if (objectInputFileName.length() != 0) {
				// Load classifier from file
				if (objectInputStream != null) {
				classifier = (Classifier) objectInputStream.readObject();
				// try and read a header (if present)
				Instances savedStructure = null;
				try {
					savedStructure = (Instances) objectInputStream.readObject();
				} catch (Exception ex) {
					// don't make a fuss
				}
				if (savedStructure != null) {
					// test for compatibility with template
				//	if (!template.equalHeaders(savedStructure)) {
					//throw new Exception("training and test set are not compatible\n"
						//+ template.equalHeadersMsg(savedStructure));
				//	}
				}
				objectInputStream.close();
				} else if (xmlInputStream != null) {
				// whether KOML is available has already been checked (objectInputStream
				// would null otherwise)!
				classifier = (Classifier) KOML.read(xmlInputStream);
				xmlInputStream.close();
				}
			}

			return classifier;
		}
		/// <summary>
		/// Persist a classifier.  Most of this code is copied from the Weka source
		/// </summary>
		/// <param name="classifier"></param>
		private static void SaveClassifier(weka.classifiers.Classifier classifier)
		{
			const string objectOutputFileName = SavedClassifier;
			// Save the classifier if an object output file is provided
			if (objectOutputFileName.length() != 0)
			{
				OutputStream os = new FileOutputStream(objectOutputFileName);
				// binary
				if (!(objectOutputFileName.endsWith(".xml") || (objectOutputFileName
					.endsWith(".koml") && KOML.isPresent())))
				{
					if (objectOutputFileName.endsWith(".gz"))
					{
						os = new GZIPOutputStream(os);
					}
					ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
					objectOutputStream.writeObject(classifier);
					//if (template != null) {
					//objectOutputStream.writeObject(template);
					//}
					objectOutputStream.flush();
					objectOutputStream.close();
				}
				// KOML/XML
				else
				{
					BufferedOutputStream xmlOutputStream = new BufferedOutputStream(os);
					if (objectOutputFileName.endsWith(".xml"))
					{
						XMLSerialization xmlSerial = new XMLClassifier();
						xmlSerial.write(xmlOutputStream, classifier);
					}
					else
						// whether KOML is present has already been checked
						// if not present -> ".koml" is interpreted as binary - see above
						if (objectOutputFileName.endsWith(".koml"))
						{
							KOML.write(xmlOutputStream, classifier);
						}
					xmlOutputStream.close();
				}
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
			return DateTime.Now.Ticks / 10000;
		}

		private static void DumpToFile(List<long> data, String fileName)
		{
			try
			{
				BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
				foreach (long value in data)
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

			return (float)result / numResults;
		}

	}
}