package de.djl.classification;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;


import java.nio.file.Path;
import java.util.Map;

public class ActivationViewer {
    private static final Logger log = LoggerFactory.getLogger(ActivationViewer.class);

    public static void visualizeLayerGrid(
            ClassificationModel cm,
            RandomAccessDataset dataset,
            int imageSize,
            int inChannels,
            String layerName,
            Path outFile,
            int tileSize) throws Exception {

        try (Trainer trainer = cm.getModel().newTrainer(new DefaultTrainingConfig(cm.getLoss()))) {
            trainer.initialize(new Shape(1, inChannels, imageSize, imageSize));
            try (Batch batch = trainer.iterateDataset(dataset).iterator().next()) {
                trainer.forward(batch.getData()); // triggers taps
                Map<String, NDArray> acts = cm.getLastActivationsSnapshot();
                NDArray feat = acts.get(layerName);
                if (feat == null) {
                    throw new IllegalArgumentException("No activation captured for '" + layerName + "'. Available: " + acts.keySet());
                }
                ImageUtils.saveFeatureGrid(feat, outFile, tileSize);
                log.info("Saved activation grid '{}' -> {} shape={}", layerName, outFile.toAbsolutePath(), feat.getShape());
            }
        }
    }

    public static void visualizeLayerFlexible(
            ClassificationModel cm,
            RandomAccessDataset dataset,
            int imageSize,
            int inChannels,
            String layerName,
            Path outFile,
            int tileSize) throws Exception {

        try (Trainer trainer = cm.getModel().newTrainer(new DefaultTrainingConfig(cm.getLoss()))) {
            trainer.initialize(new Shape(1, inChannels, imageSize, imageSize));
            try (Batch batch = trainer.iterateDataset(dataset).iterator().next()) {
                trainer.forward(batch.getData());
                Map<String, NDArray> acts = cm.getLastActivationsSnapshot();
                NDArray feat = acts.get(layerName);
                if (feat == null) {
                    throw new IllegalArgumentException("No activation captured for '" + layerName + "'. Available: " + acts.keySet());
                }
                int dim = feat.getShape().dimension();
                if (dim >= 3) {
                    ImageUtils.saveFeatureGrid(feat, outFile, tileSize);
                } else {
                    ImageUtils.saveVectorStripe(feat, outFile, Math.max(32, tileSize / 2));
                }
                log.info("Saved activation '{}' -> {} shape={}", layerName, outFile.toAbsolutePath(), feat.getShape());
            }
        }
    }

    public static void visualizeLayerFromImagePath(
            ClassificationModel cm,
            Path imagePath,
            int imageSize,
            String layerName,
            Path outFile,
            int tileSize) throws Exception {

        try (Trainer trainer = cm.getModel().newTrainer(new DefaultTrainingConfig(cm.getLoss()))) {
            System.out.println("Available taps: " + cm.getLastActivationsSnapshot().keySet());
            trainer.initialize(new Shape(1, 3, imageSize, imageSize));

            Image img = ImageFactory.getInstance().fromFile(imagePath);
            NDArray x = img.toNDArray(trainer.getManager());
            x = NDImageUtils.resize(x, imageSize, imageSize);
            x = NDImageUtils.toTensor(x);
            x = NDImageUtils.normalize(
                    x,
                    new float[]{0.485f, 0.456f, 0.406f},
                    new float[]{0.229f, 0.224f, 0.225f}
            );
            x = x.expandDims(0);

            trainer.forward(new NDList(x));
            Map<String, NDArray> acts = cm.getLastActivationsSnapshot();
            NDArray feat = acts.get(layerName);
            if (feat == null) {
                throw new IllegalArgumentException("No activation captured for '" + layerName + "'. Available: " + acts.keySet());
            }
            int dim = feat.getShape().dimension();
            if (dim >= 3) {
                ImageUtils.saveFeatureGrid(feat, outFile, tileSize);
            } else {
                ImageUtils.saveVectorStripe(feat, outFile, Math.max(32, tileSize / 2));
            }
            log.info("Saved activation from image '{}' for layer '{}' -> {} shape={}", imagePath.getFileName(), layerName, outFile.toAbsolutePath(), feat.getShape());
        }
    }
}
