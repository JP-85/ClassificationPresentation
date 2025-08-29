package de.djl.classification;

import java.util.List;

public class ModelConfig {
    public String name;
    public int stride;
    public int[] kernel;
    public int[] maxPoolSize;
    public String optimizer;
    public float learningRate;
    public int convLayers;
    public List<Integer> denseUnits;
    public String activation;
    public int batchSize;
    public double dropout;
}
