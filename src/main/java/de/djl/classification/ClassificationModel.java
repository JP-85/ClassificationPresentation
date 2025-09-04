package de.djl.classification;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * AlexNet-artiges Klassifikationsnetz mit konfigurierbaren Taps (pre/pool/fc/logits)
 * für Aktivierungsvisualisierung. Kompatibel mit DJL 0.33.
 */
public class ClassificationModel {

    public static final int DEFAULT_IMAGE_SIZE = 224;

    public Model getModel() { return model; }
    public Loss  getLoss()  { return loss;  }

    public static class History {
        public final List<Double> trainLoss = new ArrayList<>();
        public final List<Double> valLoss   = new ArrayList<>();
        public final List<Double> trainAcc  = new ArrayList<>();
        public final List<Double> valAcc    = new ArrayList<>();
        public int[][] confusion2x2 = new int[][]{{0,0},{0,0}};
        public void add(double tl, double vl, double ta, double va) {
            trainLoss.add(tl); valLoss.add(vl); trainAcc.add(ta); valAcc.add(va);
        }
    }

    private final Settings.Setting setting;
    private final Loss loss = Loss.softmaxCrossEntropyLoss();
    private final Model model;
    private final int numClasses;

    private final boolean enableTaps;
    private final NDManager snapManager;               // only used when enableTaps=true
    private final Map<String, NDArray> lastActivations; // same

    public ClassificationModel(Settings.Setting setting, int numClasses, boolean enableTaps) {
        this.setting = setting;
        this.numClasses = numClasses;
        this.enableTaps = enableTaps;
        this.model = Model.newInstance("cnn");
        if (enableTaps) {
            this.snapManager = NDManager.newBaseManager();
            this.lastActivations = new LinkedHashMap<>();
        } else {
            this.snapManager = null;
            this.lastActivations = Collections.emptyMap();
        }
        this.model.setBlock(buildBlock());
    }

    public Block buildBlock() {
        SequentialBlock net = new SequentialBlock();

        int base = setting.baseChannels != null ? setting.baseChannels : 64;
        int cap  = setting.maxChannels  != null ? setting.maxChannels  : 512;
        int outChannels = base;

        int padH = Math.max(0, setting.kernel[0] / 2);
        int padW = Math.max(0, setting.kernel[1] / 2);

        for (int i = 0; i < setting.convLayers; i++) {
            String prefix = "conv" + (i + 1);

            net.add(Conv2d.builder()
                    .setKernelShape(new Shape(setting.kernel[0], setting.kernel[1]))
                    .optStride(new Shape(setting.stride, setting.stride))
                    .optPadding(new Shape(padH, padW))
                    .setFilters(outChannels)
                    .build());
            net.add(BatchNorm.builder().build());
            net.add(activationBlock(setting.activation));

            addTap(net, prefix + "_pre");

            net.add(Pool.maxPool2dBlock(new Shape(setting.maxPoolSize[0], setting.maxPoolSize[1])));

            addTap(net, prefix + "_pool");

            addTap(net, prefix);

            outChannels = Math.min(outChannels * 2, cap);
        }

        if (Boolean.TRUE.equals(setting.globalAvgPool)) {
            net.add(Pool.globalAvgPool2dBlock());
        }
        net.add(Blocks.batchFlattenBlock());

        for (int i = 0; i < setting.denseUnits.length; i++) {
            net.add(Linear.builder().setUnits(setting.denseUnits[i]).build());
            net.add(activationBlock(setting.activation));

            addTap(net, "fc" + (i + 1));

            if (setting.dropout > 0) {
                net.add(Dropout.builder().optRate((float) setting.dropout).build());
            }
        }

        net.add(Linear.builder().setUnits(numClasses).build());
        addTap(net, "logits");
        return net;
    }

    private Block activationBlock(String name) {
        if ("leakyrelu".equalsIgnoreCase(name)) {
            float alpha = setting.leakyAlpha != null ? setting.leakyAlpha.floatValue() : 0.01f;
            return Activation.leakyReluBlock(alpha);
        }
        return Activation.reluBlock();
    }

    private void addTap(SequentialBlock net, String name) {
        if (!enableTaps) return;
        net.add(new LambdaBlock(list -> {
            try {
                NDArray a = list.head();
                int dim = a.getShape().dimension();
                if (dim >= 1) {
                    float[] data = a.toFloatArray();
                    NDArray snap = snapManager.create(data, a.getShape());
                    lastActivations.put(name, snap);
                }
            } catch (Throwable ignore) { /* taps sollen niemals forward brechen */ }
            return list;
        }));
    }

    private Optimizer makeOptimizer() {
        String opt = setting.optimizer == null ? "adam" : setting.optimizer.toLowerCase(Locale.ROOT);
        float lr = (float) setting.learningRate;
        return switch (opt) {
            case "sgd" -> Optimizer.sgd().setLearningRateTracker(Tracker.fixed(lr)).optMomentum(0.9f).build();
            case "rmsprop" -> Optimizer.rmsprop().optLearningRateTracker(Tracker.fixed(lr)).build();
            default -> Optimizer.adam().optLearningRateTracker(Tracker.fixed(lr)).build();
        };
    }

    public History fit(RandomAccessDataset train, RandomAccessDataset val, int epochs, int imageSize, int inChannels)
            throws IOException, TranslateException {
        History hist = new History();

        try (Trainer trainer = model.newTrainer(new DefaultTrainingConfig(loss).optOptimizer(makeOptimizer()))) {
            trainer.initialize(new Shape(1, inChannels, imageSize, imageSize));
            final int totalBatches = (int) Math.ceil((double) train.size() / Math.max(1, setting.batchSize));

            for (int epoch = 0; epoch < epochs; epoch++) {
                System.out.printf("%nEpoch %d/%d%n", epoch + 1, epochs);

                double sumLossT = 0.0; long nT = 0; long correctT = 0; int batchCount = 0;

                for (Batch batch : trainer.iterateDataset(train)) {
                    NDArray preds; NDArray L;
                    try (GradientCollector gc = trainer.newGradientCollector()) {
                        preds = trainer.forward(batch.getData()).getFirst(); // (N,K)

                        NDArray y = batch.getLabels().head().squeeze();
                        if (y.getShape().dimension() == 0) {
                            y = y.expandDims(0);
                        }
                        if (y.getShape().dimension() > 1) {
                            y = y.reshape(new Shape(y.size()));
                        }
                        y = y.toType(DataType.INT64, false);

                        NDArray Larr = loss.evaluate(new NDList(y), new NDList(preds)).get(0);
                        L = (Larr.getShape().dimension() == 0) ? Larr : Larr.mean();
                        gc.backward(L);
                    }
                    trainer.step();

                    long bs = preds.getShape().get(0);
                    sumLossT += L.getFloat() * bs;
                    nT += bs;

                    NDArray labels = batch.getLabels().head().toType(DataType.INT64, false);
                    long correctBatch = preds.argMax(1).eq(labels).toType(DataType.INT64, false).sum().getLong();
                    correctT += correctBatch;

                    batch.close();

                    batchCount++;
                    double avgLoss = sumLossT / Math.max(1, nT);
                    double acc = nT == 0 ? 0.0 : (double) correctT / nT;
                    printProgressBar(batchCount, totalBatches, avgLoss, acc);
                }
                System.out.println();

                double trainLoss = sumLossT / Math.max(1, nT);
                double trainAcc = nT == 0 ? 0.0 : (double) correctT / nT;

                double sumLossV = 0.0; long nV = 0; long correctV = 0;
                int[][] cm = new int[][]{{0,0},{0,0}};

                for (Batch batch : trainer.iterateDataset(val)) {
                    NDArray preds = trainer.forward(batch.getData()).getFirst();

                    NDArray y = batch.getLabels().head().squeeze();
                    if (y.getShape().dimension() == 0) y = y.expandDims(0);
                    if (y.getShape().dimension() > 1) y = y.reshape(new Shape(y.size()));
                    y = y.toType(DataType.INT64, false);

                    NDArray L = loss.evaluate(new NDList(y), new NDList(preds)).get(0);
                    if (L.getShape().dimension() != 0) L = L.mean();

                    long bs = preds.getShape().get(0);
                    sumLossV += L.getFloat() * bs;
                    nV += bs;

                    correctV += preds.argMax(1).eq(y).toType(DataType.INT64, false).sum().getLong();

                    if (numClasses == 2) {
                        long[] p = preds.argMax(1).toLongArray();
                        long[] yy = y.toLongArray();
                        for (int i = 0; i < p.length; i++) {
                            int yi = (int) yy[i], pi = (int) p[i];
                            if (yi>=0 && yi<2 && pi>=0 && pi<2) cm[yi][pi]++;
                        }
                    }
                    batch.close();
                }

                double valLoss = sumLossV / Math.max(1, nV);
                double valAcc = nV == 0 ? 0.0 : (double) correctV / nV;
                hist.confusion2x2 = cm;
                hist.add(trainLoss, valLoss, trainAcc, valAcc);

                System.out.printf("→ train: loss=%.4f acc=%.2f%%   |   val: loss=%.4f acc=%.2f%%%n",
                        trainLoss, trainAcc * 100.0, valLoss, valAcc * 100.0);
            }
        }
        return hist;
    }

    public void save(Path outputDir, List<String> synset) throws IOException {
        Files.createDirectories(outputDir);
        String time = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
        Path dir = outputDir.resolve(setting.name + "-" + time);
        Files.createDirectories(dir);
        model.setProperty("classes", String.join(",", synset));
        model.save(dir, "cnn");
        Files.writeString(dir.resolve("synset.txt"), String.join("\n", synset));
    }

    public Map<String, NDArray> getLastActivationsSnapshot() {
        if (!enableTaps || lastActivations.isEmpty()) return Collections.emptyMap();
        return new LinkedHashMap<>(lastActivations);
    }

    private void printProgressBar(int current, int total, double loss, double acc) {
        int barLength = 40;
        int filled = (int) Math.round((current / (double) total) * barLength);
        if (filled > barLength) filled = barLength;
        String bar = "=".repeat(filled) + " ".repeat(barLength - filled);
        String lossStr = String.format("%sloss=%.4f%s", ansi(31), loss, ansi(0));
        String accStr  = String.format("%sacc=%.2f%%%s", ansi(32), acc * 100.0, ansi(0));
        System.out.printf("\r[%s] %d/%d  %s  %s", bar, current, total, lossStr, accStr);
        System.out.flush();
    }

    private String ansi(int code) { return "\u001B[" + code + "m"; }
}
