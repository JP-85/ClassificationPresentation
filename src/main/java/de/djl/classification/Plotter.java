// =====================
// File: src/main/java/de/djl/classification/Plotter.java
// =====================
package de.djl.classification;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Plots loss & accuracy over epochs.
 * - Expects accuracy in [0,1].
 * - Writes two PNGs: "<base>_loss.png" and "<base>_accuracy.png".
 */
public class Plotter {

    public static Path saveLossAcc(
            List<Double> lossTrain,
            List<Double> lossVal,
            List<Double> accTrain,
            List<Double> accVal,
            Path outDir,
            String baseName) throws IOException {

        Files.createDirectories(outDir);

        int n = Math.min(Math.min(lossTrain.size(), lossVal.size()), Math.min(accTrain.size(), accVal.size()));
        if (n <= 0) {
            // Nothing to plot; still return the directory
            return outDir;
        }

        double[] x = IntStream.rangeClosed(1, n).asDoubleStream().toArray();

        // --- Loss chart ---
        XYChart lossChart = new XYChartBuilder()
                .width(800).height(500)
                .title("Loss pro Epoche")
                .xAxisTitle("Epoch").yAxisTitle("Loss")
                .build();
        lossChart.addSeries("train", x, lossTrain.stream().limit(n).mapToDouble(Double::doubleValue).toArray());
        lossChart.addSeries("val",   x, lossVal.stream().limit(n).mapToDouble(Double::doubleValue).toArray());
        Path lossPng = outDir.resolve(baseName + "_loss.png");
        BitmapEncoder.saveBitmap(lossChart, lossPng.toString(), BitmapEncoder.BitmapFormat.PNG);

        // --- Accuracy chart --- (values in [0,1])
        XYChart accChart = new XYChartBuilder()
                .width(800).height(500)
                .title("Accuracy pro Epoche")
                .xAxisTitle("Epoch").yAxisTitle("Accuracy")
                .build();
        accChart.addSeries("train", x, accTrain.stream().limit(n).mapToDouble(Double::doubleValue).toArray());
        accChart.addSeries("val",   x, accVal.stream().limit(n).mapToDouble(Double::doubleValue).toArray());
        Path accPng = outDir.resolve(baseName + "_accuracy.png");
        BitmapEncoder.saveBitmap(accChart, accPng.toString(), BitmapEncoder.BitmapFormat.PNG);

        return outDir;
    }
}
