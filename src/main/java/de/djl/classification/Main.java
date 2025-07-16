package de.djl.classification;

import ai.djl.Application;

import java.io.IOException;


public class Main {
    public static void main(String[] args) throws IOException {
//        Application application = Application.CV.IMAGE_CLASSIFICATION;
//        ClassificationModel model = new ClassificationModel();
//        System.out.println(model.block);
        Preprocessing prep = new GrayscalePreprocessing(100, 100);
        CNNDataset catdog = prep.run("CatDogData", "data/PetImages");
        catdog.writeData("output/catdog100px.ser");
    }
}