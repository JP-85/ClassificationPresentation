package de.djl.classification;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.*;
import ai.djl.training.tracker.Tracker;

//import ai.djl.training.optimizer.Adam;
//import ai.djl.training.optimizer.Sgd;
//import ai.djl.training.optimizer.RmsProp;
//import ai.djl.training.optimizer.Optimizer;

import java.util.Arrays;
import java.util.function.Function;

public class CustomModel {

    private final ModelConfig config;
    private final DatasetMetadata metadata;
    private Model model;

    public CustomModel(ModelConfig config, DatasetMetadata metadata) {
        this.config = config;
        this.metadata = metadata;
        this.model = buildModel();
    }

    private Model buildModel() {
        SequentialBlock block = new SequentialBlock();

        // Convolutional Layers
        for (int i = 0; i < config.convLayers; i++) {
            block.add(Conv2d.builder()
                            .setKernelShape(new ai.djl.ndarray.types.Shape(toLong(config.kernel)))
                            .optStride(new ai.djl.ndarray.types.Shape(config.stride, config.stride))
                            .setFilters(32 * (i + 1))
                            .build())
                    .add(BatchNorm.builder().build());
            block.add(getActivation(config.activation));

            block.add(Pool.maxPool2dBlock(new ai.djl.ndarray.types.Shape(toLong(config.maxPoolSize))));
        }

        // Flatten
        block.add(ai.djl.nn.Blocks.batchFlattenBlock());

        // Dense Layers
        for (Integer units : config.denseUnits) {
            block.add(Linear.builder().setUnits(units).build())
                    .add(Activation::relu);
        }

        // Output Layer
        block.add(Linear.builder().setUnits(metadata.getSynset().size()).build());

        // Model-Objekt
        Model model = Model.newInstance(config.name);
        model.setBlock(block);
        return model;
    }

    public Trainer setupTrainer() {
        DefaultTrainingConfig trainConfig = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addTrainingListeners(TrainingListener.Defaults.logging());

        // Optimizer auswählen
        Optimizer optimizer = buildOptimizer(config);

        trainConfig.optOptimizer(optimizer);
        return model.newTrainer(trainConfig);
    }

    public Model getModel() {
        return model;
    }

    private static Function<NDList, NDList> getActivation(String name) {
        float alpha_slope_for_negatives = 0.01f;

        return switch (name.toLowerCase()) {
            case "relu" -> Activation::relu;
            case "sigmoid" -> Activation::sigmoid;
            case "tanh" -> Activation::tanh;
            case "leakyrelu" -> arr -> Activation.leakyRelu(arr, alpha_slope_for_negatives);
            default -> throw new IllegalArgumentException("Unknown activation: " + name);
        };
    }

    private static long[] toLong(int[] dims) {
        return Arrays.stream(dims)
                .mapToLong(i -> i)
                .toArray();
    }

    private Optimizer buildOptimizer(ModelConfig config) {
        Tracker lrt = Tracker.fixed(config.learningRate);

        return switch (config.optimizer.toLowerCase()) {
            case "adam" -> Optimizer.adam()
                    .optLearningRateTracker(lrt)
                    .build();
            case "sgd" -> Optimizer.sgd()
                    .setLearningRateTracker(lrt)
                    .build();
            case "rmsprop" -> Optimizer.rmsprop()
                    .optLearningRateTracker(lrt)
                    .build();
            default -> throw new IllegalArgumentException("Unbekannter Optimizer: " + config.optimizer);
        };
    }
}
