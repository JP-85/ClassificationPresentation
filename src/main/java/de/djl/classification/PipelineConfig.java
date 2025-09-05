
package de.djl.classification;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;


import java.io.InputStream;
import java.util.List;

public class PipelineConfig {
    public String setting = "baseline";
    public String settingsJson = "settings.json";

    public String raw = "data/raw/PetImages";
    public String datasetsRoot = "data/datasets";
    public double valSplit = 0.2;
    public long seed = 42L;

    public int epochs = 3;
    public int imageSize = ClassificationModel.DEFAULT_IMAGE_SIZE;
    public boolean grayscale = false;
    public boolean shuffleTrain = true;

    public boolean zoo = false;
    public String zooBackbone = "resnet";

    public boolean saveActivations = false;

    public List<String> vizLayers;

    public static PipelineConfig loadFromResources(String resource) {
        try (InputStream is = PipelineConfig.class.getClassLoader().getResourceAsStream(resource)) {
            if (is == null) return new PipelineConfig();
            ObjectMapper om = new ObjectMapper();
            return om.readValue(is, new TypeReference<>() {
            });
        } catch (Exception e) {
            return new PipelineConfig();
        }
    }

    public void applyOverrides(String[] argv) {
        for (int i=0; i<argv.length; i++) {
            switch (argv[i]) {
                case "--setting" -> this.setting = argv[++i];
                case "--settings-json" -> this.settingsJson = argv[++i];
                case "--raw" -> this.raw = argv[++i];
                case "--datasets-root" -> this.datasetsRoot = argv[++i];
                case "--val" -> this.valSplit = Double.parseDouble(argv[++i]);
                case "--seed" -> this.seed = Long.parseLong(argv[++i]);
                case "--epochs" -> this.epochs = Integer.parseInt(argv[++i]);
                case "--img" -> this.imageSize = Integer.parseInt(argv[++i]);
                case "--grayscale" -> this.grayscale = Boolean.parseBoolean(argv[++i]);
                case "--shuffle-train" -> this.shuffleTrain = Boolean.parseBoolean(argv[++i]);
                case "--save-activations" -> this.saveActivations = Boolean.parseBoolean(argv[++i]);
                default -> { }
            }
        }
    }
}
