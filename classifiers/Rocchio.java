package ir.classifiers;

import java.util.*;
import java.io.*;

import ir.vsr.*;
import ir.utilities.*;

public class Rocchio extends Classifier{
    public int numCategories;
    public boolean neg;
    public HashMap<Integer, HashMapVector> prototype;
    public InvertedIndex index;  
    boolean hasTraining;

    // initiliaze most variables 
    public Rocchio (String[] categories, List<Example> examples, boolean neg) {
        this.categories = categories;
        numCategories = categories.length;
        this.neg = neg;
        prototype = new HashMap<Integer, HashMapVector>();

         // Set up the inverted Index
        index = new InvertedIndex(examples);
        hasTraining = false;
    }

    public void train(List<Example> trainingExamples){
        // Decides whether there is training data to use or not
        if (!trainingExamples.isEmpty())
            hasTraining = true;
        else 
            hasTraining = false;

        // Initialize prototypes
        prototype = new HashMap<Integer, HashMapVector>();
        for (int x = 0; x < numCategories; x++){
            HashMapVector vector = new HashMapVector();
            prototype.put(x, vector);
        }

        // Add training data to create each prototype
        for (Example e : trainingExamples){
            HashMapVector proto = prototype.get(e.getCategory());
            HashMapVector adder = getTFIDF(e);
            proto.add(adder);
        }

        // if -neg then also remove data from categories of a different type
        if (neg){
            //Parse through the prototypes
            for (Map.Entry<Integer, HashMapVector> entry: prototype.entrySet()){
                // Parse through trainingExamples
                for (Example e : trainingExamples){
                    // Do not remove if it is in the same category
                    if (entry.getKey() != e.getCategory()){
                        HashMapVector sub = getTFIDF(e);
                        entry.getValue().subtract(sub);
                    }
                }
            }
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
        HashMapVector temp = getTFIDF(testExample);
        double m = -2.0;
        double s = 0.0;

        // default guess
        int guess = 0;

        // calculate cosine similarity with each prototype
        HashMapVector protoVec = new HashMapVector();
        for (int x = 0; x < numCategories; x++){
            protoVec = prototype.get(x);
            s = temp.cosineTo(protoVec);
            // if calculates cosine similarity is greater than current guess
            if (s > m){
                m = s;
                // updated guess
                guess = x;
            }
        }

        // check guess
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
		return "Rocchio";
    }
    
}
