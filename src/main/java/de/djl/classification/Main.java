package de.djl.classification;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
    private static final Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws Exception {
        String configResource = "runconfig.json";

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
