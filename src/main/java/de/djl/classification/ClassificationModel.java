package de.djl.classification;

import ai.djl.*;
import ai.djl.nn.*;
import ai.djl.nn.core.*;
import ai.djl.training.*;
import java.nio.file.*;

import ai.djl.*;
import ai.djl.ndarray.types.*;
import ai.djl.training.*;
import ai.djl.training.dataset.*;
import ai.djl.training.initializer.*;
import ai.djl.training.loss.*;
import ai.djl.training.listener.*;
import ai.djl.training.evaluator.*;
import ai.djl.training.optimizer.*;
import ai.djl.training.util.*;
import ai.djl.basicmodelzoo.cv.classification.*;
import ai.djl.basicmodelzoo.basic.*;
import ai.djl.basicdataset.cv.classification.*;
import ai.djl.ndarray.NDManager;
import org.apache.commons.lang3.ArrayUtils;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.pooling.Pool;
//import ai.djl.nn.core.Dropout;
import ai.djl.nn.norm.Dropout;

public class ClassificationModel {
    public SequentialBlock block;
    long inputSize;
    long outputSize;

    public ClassificationModel() {
        this(28*28, 10);
    }

    public ClassificationModel(long inputSize, long outputSize) {
//        this.inputSize = inputSize;
//        this.outputSize = outputSize;
//        this.block = new SequentialBlock();
//        this.block.add(Blocks.batchFlattenBlock(this.inputSize));
//        this.block.add(Linear.builder().setUnits(128).build());
//        this.block.add(Activation::relu);
//        this.block.add(Linear.builder().setUnits(64).build());
//        this.block.add(Activation::relu);
//        this.block.add(Linear.builder().setUnits(this.outputSize).build());


//        NDManager manager = NDManager.newBaseManager();

        // Here, we use a larger 11 x 11 window to capture objects. At the same time,
        // we use a stride of 4 to greatly reduce the height and width of the output.
        //Here, the number of output channels is much larger than that in LeNet
        this.block = new  SequentialBlock();
        this.block
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(11, 11))
                        .optStride(new Shape(4, 4))
                        .setFilters(96).build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                // Make the convolution window smaller, set padding to 2 for consistent
                // height and width across the input and output, and increase the
                // number of output channels
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(5, 5))
                        .optPadding(new Shape(2, 2))
                        .setFilters(256).build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                // Use three successive convolutional layers and a smaller convolution
                // window. Except for the final convolutional layer, the number of
                // output channels is further increased. Pooling layers are not used to
                // reduce the height and width of input after the first two
                // convolutional layers
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .setFilters(384).build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .setFilters(384).build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .setFilters(256).build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                // Here, the number of outputs of the fully connected layer is several
                // times larger than that in LeNet. Use the dropout layer to mitigate
                // overfitting
                .add(Blocks.batchFlattenBlock())
                .add(Linear
                        .builder()
                        .setUnits(4096)
                        .build())
                .add(Activation::relu)
                .add(Dropout
                        .builder()
                        .optRate(0.5f)
                        .build())
                .add(Linear
                        .builder()
                        .setUnits(4096)
                        .build())
                .add(Activation::relu)
                .add(Dropout
                        .builder()
                        .optRate(0.5f)
                        .build())
                // Output layer. Since we are using Fashion-MNIST, the number of
                // classes is 10, instead of 1000 as in the paper
                .add(Linear.builder().setUnits(10).build());
    }

//    public fit()
}
