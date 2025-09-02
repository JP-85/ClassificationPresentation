// =====================
// File: src/main/java/de/djl/classification/Main.java
// =====================
package de.djl.classification;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
    private static final Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws Exception {
        // Choose config resource by commenting
        String configResource = "runconfig.json";            // training pipeline
        // String configResource = "configs/baseline_small.json";
        // String configResource = "configs/fast_demo.json";

        // To run a model-zoo demo quickly set `zoo=true` in the chosen config,
        // or pass --zoo true on CLI.

        log.info("Loading pipeline config from resource: {}", configResource);
        PipelineConfig cfg = PipelineConfig.loadFromResources(configResource);
        cfg.applyOverrides(args);

        if (cfg.zoo) {
            log.info("Running model-zoo demo with backbone: {}", cfg.zooBackbone);
            CNNPipeline.runZoo(cfg);
        } else {
            CNNPipeline.run(cfg);
        }
    }
}
