package ir.classifiers;

import java.util.*;
import java.io.*;

import ir.vsr.*;
import ir.utilities.*;
import java.util.Random;

public class KNN extends Classifier{
    public int k;
    public int numCategories;
    public InvertedIndex index;
    public HashMap<Example, HashMapVector> vectors;
    boolean hasTraining;

    // initiliaze most variables 
    public KNN (String[] categories, List<Example> examples, int k) {
        this.k = k;
        this.categories = categories;
        numCategories = categories.length;
        vectors = new HashMap<Example, HashMapVector>();

        // Set up the inverted Index
        index = new InvertedIndex(examples);
        hasTraining = false;
    }

    public void train (List<Example> trainingExamples){
        // Decides whether there is training data to use or not
        if (!trainingExamples.isEmpty())
            hasTraining = true;
        else 
            hasTraining = false;

        // Create a new map of vectors for each new set of trainingExamples
        vectors = new HashMap<Example, HashMapVector>();

        // For each example, get the TF-IDF weighted hashmapvector and put in hashmap
        for (Example e: trainingExamples){
            HashMapVector vector = getTFIDF(e);
            vectors.put(e, vector);
        }
    }

    public boolean test (Example testExample){
        // If there are no training examples, pick a random category
        Random rand = new Random();
        if (!hasTraining){
            int guess = rand.nextInt(numCategories); 
            boolean correct = (guess == testExample.getCategory());
            return correct;
        }
        // get TF-IDF weighted vector for the test
        HashMapVector vector = getTFIDF(testExample);

        // Data structure to hold all cosine similarity results
        TreeMap<Double, Integer> categoryData = new TreeMap<Double, Integer>();

        // Compute cosine similarity between test and all training exampels and store it
        for (Map.Entry<Example, HashMapVector> entry : vectors.entrySet()){
            double sim = 1-vector.cosineTo(entry.getValue());
            categoryData.put(sim, entry.getKey().getCategory());
        }

        // Find out which category the test belongs to
        double[] occurence = new double[categories.length];
        int index = 0;

        // Go through the most similiar training examples and find which category majority belong to
        for (Map.Entry<Double, Integer> entry : categoryData.entrySet()){
            for (int x = 0; x < k; x++){
                occurence[entry.getValue()] += 1;
            }
        }

        // Guess and check
        int guess = argMax(occurence);
        boolean correct = (guess == testExample.getCategory());
        return correct;
    }

    // Recieves an example and returns the TF-IDF weighted version
    public HashMapVector getTFIDF (Example e) {
        HashMapVector eVector = e.getHashMapVector(); // example HashMapVector
        HashMapVector temp = new HashMapVector(); // HashMapVector to return
        // normalize by maxFreq
        double maxFreq = eVector.maxWeight();

        for (Map.Entry<String, Weight> entry : eVector.entrySet()){
            String token = entry.getKey();

            double weight = entry.getValue().getValue(); // get TF info
            double idf = index.tokenHash.get(token).idf; // Get IDF from invertedIndex
            double tf_idf = weight/maxFreq*idf; // compute new weight with normalization

            // Set new weight
            Weight newWeight = new Weight();
            newWeight.setValue(tf_idf);
            
            temp.hashMap.put(token, newWeight);
        }

        // returns weighted version
        return temp;
    }

    public String getName(){
		return "KNN";
	}
    
}