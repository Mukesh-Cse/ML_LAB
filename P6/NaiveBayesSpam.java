import java.io.*;
import java.util.*;

public class NaiveBayesSpam {

    static class Doc {
        String text;
        String label;

        Doc(String t, String l) {
            text = t;
            label = l;
        }
    }

    static Map<String, Map<String, Integer>> wordCounts = new HashMap<>();
    static Map<String, Integer> classCounts = new HashMap<>();
    static Set<String> vocabulary = new HashSet<>();
    static int totalDocs;

    // -----------------------------
    // TRAINING
    // -----------------------------
    static void train(List<Doc> docs) {
        totalDocs = docs.size();

        for (Doc d : docs) {
            classCounts.put(d.label,
                    classCounts.getOrDefault(d.label, 0) + 1);

            wordCounts.putIfAbsent(d.label, new HashMap<>());

            String[] words = d.text.toLowerCase().split("\\s+");

            for (String w : words) {
                vocabulary.add(w);
                Map<String, Integer> wc = wordCounts.get(d.label);
                wc.put(w, wc.getOrDefault(w, 0) + 1);
            }
        }
    }

    // -----------------------------
    // PREDICTION
    // -----------------------------
    static String predict(String text) {
        String[] words = text.toLowerCase().split("\\s+");

        double bestScore = Double.NEGATIVE_INFINITY;
        String bestClass = null;

        for (String label : classCounts.keySet()) {

            double logProb = Math.log(
                    (double) classCounts.get(label) / totalDocs);

            Map<String, Integer> wc = wordCounts.get(label);
            int totalWords = wc.values().stream().mapToInt(i -> i).sum();

            for (String w : words) {
                int count = wc.getOrDefault(w, 0);

                double prob = (count + 1.0) /
                        (totalWords + vocabulary.size());

                logProb += Math.log(prob);
            }

            if (logProb > bestScore) {
                bestScore = logProb;
                bestClass = label;
            }
        }
        return bestClass;
    }

    // -----------------------------
    // MAIN
    // -----------------------------
    public static void main(String[] args) throws Exception {

        BufferedReader br = new BufferedReader(
                new FileReader("6aspam1000.csv"));

        List<Doc> docs = new ArrayList<>();
        String line;

        br.readLine(); // skip header

        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",", 2);
            docs.add(new Doc(parts[1], parts[0]));
        }
        br.close();

        // Split 70% train, 30% test
        int split = (int) (docs.size() * 0.7);
        List<Doc> train = docs.subList(0, split);
        List<Doc> test = docs.subList(split, docs.size());

        train(train);

        int correct = 0;
        int tp = 0, fp = 0, fn = 0;

        System.out.println("Test Results:\n");

        for (Doc d : test) {
            String pred = predict(d.text);

            System.out.println("Message : " + d.text);
            System.out.println("Actual  : " + d.label);
            System.out.println("Predicted: " + pred);
            System.out.println("---------------------------");

            if (pred.equals(d.label))
                correct++;

            if (pred.equals("spam") && d.label.equals("spam"))
                tp++;
            else if (pred.equals("spam") && d.label.equals("ham"))
                fp++;
            else if (pred.equals("ham") && d.label.equals("spam"))
                fn++;
        }

        double accuracy = (double) correct / test.size();
        double precision = tp / (double) (tp + fp + 1e-6);
        double recall = tp / (double) (tp + fn + 1e-6);
        double f1 = 2 * precision * recall / (precision + recall + 1e-6);

        System.out.println("\nEvaluation Metrics:");
        System.out.println("Accuracy : " + accuracy);
        System.out.println("Precision: " + precision);
        System.out.println("Recall   : " + recall);
        System.out.println("F1 Score : " + f1);
    }
}
