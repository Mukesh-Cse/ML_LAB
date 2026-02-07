import java.io.*;
import java.util.*;

public class NaiveBayesBBC {

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
    // PREDICT
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
                new FileReader("6bBBC1000.csv"));

        List<Doc> docs = new ArrayList<>();
        String line;

        br.readLine(); // skip header

        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",", 2);
            docs.add(new Doc(parts[1], parts[0]));
        }
        br.close();

        // 70% train, 30% test
        int split = (int) (docs.size() * 0.7);
        List<Doc> train = docs.subList(0, split);
        List<Doc> test = docs.subList(split, docs.size());

        train(train);

        int correct = 0;

        // For metrics (multi-class weighted)
        Map<String, Integer> tp = new HashMap<>();
        Map<String, Integer> fp = new HashMap<>();
        Map<String, Integer> fn = new HashMap<>();

        System.out.println("Test Results:\n");

        for (Doc d : test) {
            String pred = predict(d.text);

            System.out.println("Text     : " + d.text);
            System.out.println("Actual   : " + d.label);
            System.out.println("Predicted: " + pred);
            System.out.println("---------------------------");

            if (pred.equals(d.label))
                correct++;

            if (pred.equals(d.label)) {
                tp.put(pred, tp.getOrDefault(pred, 0) + 1);
            } else {
                fp.put(pred, fp.getOrDefault(pred, 0) + 1);
                fn.put(d.label, fn.getOrDefault(d.label, 0) + 1);
            }
        }

        double accuracy = (double) correct / test.size();

        double precisionSum = 0;
        double recallSum = 0;
        int classes = classCounts.size();

        for (String c : classCounts.keySet()) {
            double t = tp.getOrDefault(c, 0);
            double f_p = fp.getOrDefault(c, 0);
            double f_n = fn.getOrDefault(c, 0);

            double precision = t / (t + f_p + 1e-6);
            double recall = t / (t + f_n + 1e-6);

            precisionSum += precision;
            recallSum += recall;
        }

        double precision = precisionSum / classes;
        double recall = recallSum / classes;
        double f1 = 2 * precision * recall / (precision + recall + 1e-6);

        System.out.println("\nEvaluation Metrics:");
        System.out.println("Accuracy : " + accuracy);
        System.out.println("Precision: " + precision);
        System.out.println("Recall   : " + recall);
        System.out.println("F1 Score : " + f1);
    }
}
