package ir.classifiers;

import java.util.*;

/**
 * Wrapper class to test NaiveBayes classifier using 10-fold CV.
 * Running it with -debug option gives very detailed output
 *
 * @author Sugato Basu
 */

public class TestRocchio {
  /**
   * A driver method for testing the NaiveBayes classifier using
   * 10-fold cross validation.
   *
   * @param args a list of command-line arguments.  Specifying "-debug"
   *             will provide detailed output
   */
  public static void main(String args[]) throws Exception {
    String dirName = "/u/mooney/ir-code/corpora/curlie-science/";
    String[] categories = {"bio", "chem", "phys"};
    System.out.println("Loading Examples from " + dirName + "...");
    List<Example> examples = new DirectoryExamplesConstructor(dirName, categories).getExamples();
    System.out.println("Initializing Rocchio classifier...");
    Rocchio RC;

    boolean neg;
    // Find out whether to use the modified version or not
    if (args.length > 0 && args[0].equals("-neg"))
        neg = true;
    else
        neg = false;
    RC = new Rocchio(categories, examples, neg);

    // Perform 10-fold cross validation to generate learning curve
    CVLearningCurve cvCurve = new CVLearningCurve(RC, examples);
    cvCurve.run();
  }
}
