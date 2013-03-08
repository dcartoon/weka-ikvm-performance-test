weka-ikvm-performance-test
==========================

Basic project to evaluate the performance of [http://www.cs.waikato.ac.nz/ml/weka/](Weka) running under the JVM versus under the CLR(using IKVM).  This code is designed to run on 

## Setup - Java

You will need to have the Weka Jar somewhere on your system already. If using Eclipse, update the .classpath accordingly for the jar location.  Otherwise, update the classpath in compile.bat and run.bat.

## Running - Java

After updating the classpath, you can use run.bat to run the Weka on the JVM.  This is an implementation of the code in the [IKVM Tutorial](http://weka.wikispaces.com/IKVM+with+Weka+tutorial) with some additional timing calls.  The code will print average times for the training/classification stages and dump the times to CSVs for more advanced analysis.

## Setup - IKVM/C#
You will need to download [IKVM](http://sourceforge.net/projects/ikvm/files/) and run it on the Weka Jar from your Weka installation.

For example:
ikvmc -target:library "c:\Program Files\Weka-3-6\weka.jar" -out:weka.dll

The solution is a Visual Studio 2012 solution.  You will most likely need to update the reference locations for weka.dll(from the previous step), and the IKVM.OpenJDK.*.dll and IKVM.Runtime.dll (these are in the bin folder under your IKVM installation directory).

## Running - IKVM
The code assumes that it will be run from the bin\Release directory

## Performance

Testing was done on a Core i7-2600 - 3.4 GHz machine running Windows 7 and Java 1.7.0 - 64 bit.  Timed using a 90% split of data for training/test.

Classifier				|	Operation	|	Time(Java)	|	Time(C#)
-----------				|	---------	|	----		|	--------	
J48						|	Training	|	17 s		|	65 s
J48						|	Test		|	6 ms		|	3.5 ms
RandomForest(10 trees)	|	Training	|	35 s		|	138 s
RandomForest(10 trees)	|	Test		|	20 ms		|	41 ms
RandomForest(20 trees)	|	Training	|	72 s		|	262 s
RandomForest(20 trees)	|	Test		|	56 ms		|	86 ms
RandomForest(50 trees)	|	Training	|	172 s		|	653 s
RandomForest(50 trees)	|	Test		|	133 ms		|	238 ms
RandomForest(75 trees)	|	Training	|	256 s		|	961 s
RandomForest(75 trees)	|	Test		|	214 ms		|	376 ms