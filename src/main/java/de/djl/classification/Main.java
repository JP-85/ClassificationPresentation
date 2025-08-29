package de.djl.classification;

public class Main {
    public static void main(String[] args) {
        // Erstellen eines neuen Datensatzes
        CNNPipeline pipeline = CNNPipeline.builder()
                .addPreprocessing("PetImages",
                        "cats_dogs_64",
                        64,
                        64,
                        true,
                        false);
        pipeline.run();
        pipeline.close();

        System.out.println("\n------------------------------------------------\n");

        // Beispiel 2: Laden eines bestehenden Datensatzes
        CNNPipeline testPipeline = CNNPipeline.builder()
                .addPreprocessing("cats_dogs_64");
        testPipeline.run();

        System.out.println(testPipeline.getDataSet());
        System.out.println(testPipeline.getDataSet().getMetadata());

        testPipeline.close();
    }
}