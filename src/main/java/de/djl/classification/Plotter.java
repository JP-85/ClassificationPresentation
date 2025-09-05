package de.djl.classification;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.IntStream;

public class Plotter {

    public static void saveLossAcc(
            List<Double> lossTrain,
            List<Double> lossVal,
            List<Double> accTrain,
            List<Double> accVal,
            Path outDir,
            String baseName) throws IOException {

        Files.createDirectories(outDir);

        int n = Math.min(Math.min(lossTrain.size(), lossVal.size()), Math.min(accTrain.size(), accVal.size()));
        if (n <= 0) {
            return;
        }

        double[] x = IntStream.rangeClosed(1, n).asDoubleStream().toArray();

        XYChart lossChart = new XYChartBuilder()
                .width(800).height(500)
                .title("Loss pro Epoche")
                .xAxisTitle("Epoch").yAxisTitle("Loss")
                .build();
        lossChart.addSeries("train", x, lossTrain.stream().limit(n).mapToDouble(Double::doubleValue).toArray());
        lossChart.addSeries("val",   x, lossVal.stream().limit(n).mapToDouble(Double::doubleValue).toArray());
        Path lossPng = outDir.resolve(baseName + "_loss.png");
        BitmapEncoder.saveBitmap(lossChart, lossPng.toString(), BitmapEncoder.BitmapFormat.PNG);

        XYChart accChart = new XYChartBuilder()
                .width(800).height(500)
                .title("Accuracy pro Epoche")
                .xAxisTitle("Epoch").yAxisTitle("Accuracy")
                .build();
        accChart.addSeries("train", x, accTrain.stream().limit(n).mapToDouble(Double::doubleValue).toArray());
        accChart.addSeries("val",   x, accVal.stream().limit(n).mapToDouble(Double::doubleValue).toArray());
        Path accPng = outDir.resolve(baseName + "_accuracy.png");
        BitmapEncoder.saveBitmap(accChart, accPng.toString(), BitmapEncoder.BitmapFormat.PNG);

    }
}
