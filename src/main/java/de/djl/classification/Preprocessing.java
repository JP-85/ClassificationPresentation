package de.djl.classification;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.imgscalr.Scalr;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

public class Preprocessing {
    private static final Logger log = LoggerFactory.getLogger(Preprocessing.class);

    public static class Metadata {
        public List<String> classes;
        public Map<String, Integer> trainCount;
        public Map<String, Integer> valCount;
        public int targetSize;
        public boolean grayscaleAppearance;
        public List<String> skipped = new ArrayList<>();
    }

    public record PreparedPaths(Path outRoot, Path trainRoot, Path valRoot, Path metadataJson) { }

    private static final Set<String> ALLOWED_EXT = Set.of(
            ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"
    );

    public static PreparedPaths prepareDatasets(
            Path rawRoot,
            Path datasetsRoot,
            double valSplit,
            long seed,
            String runName,
            int targetSize,
            boolean grayscaleAppearance
    ) throws IOException {
        if (!Files.isDirectory(rawRoot)) {
            throw new IOException("Raw root not found: " + rawRoot);
        }

        Path outRoot = datasetsRoot.resolve(runName);
        Path trainRoot = outRoot.resolve("train");
        Path valRoot = outRoot.resolve("val");
        Files.createDirectories(trainRoot);
        Files.createDirectories(valRoot);

        List<Path> classDirs;
        try (var stream = Files.list(rawRoot)) {
            classDirs = stream.filter(Files::isDirectory).sorted().toList();
        }
        if (classDirs.isEmpty()) throw new IOException("No class folders under " + rawRoot);

        Random rnd = new Random(seed);
        Map<String,Integer> trainCount = new LinkedHashMap<>();
        Map<String,Integer> valCount = new LinkedHashMap<>();
        List<String> skipped = new ArrayList<>();

        for (Path clsDir : classDirs) {
            String cls = clsDir.getFileName().toString();
            List<Path> images;
            try (var walk = Files.walk(clsDir)) {
                images = walk.filter(Files::isRegularFile)
                        .filter(Preprocessing::hasAllowedExt)
                        .filter(p -> isSupportedImage(p, skipped))
                        .collect(Collectors.toList());
            }
            Collections.shuffle(images, rnd);

            int n = images.size();
            int nVal = Math.max(1, (int) Math.round(n * valSplit));
            int nTrain = n - nVal;

            Path tOut = trainRoot.resolve(cls); Files.createDirectories(tOut);
            Path vOut = valRoot.resolve(cls);  Files.createDirectories(vOut);

            for (int i = 0; i < n; i++) {
                Path src = images.get(i);
                Path dst = (i < nTrain ? tOut : vOut).resolve(src.getFileName().toString());
                try {
                    transformAndSave(src, dst, targetSize, grayscaleAppearance);
                } catch (Exception ex) {
                    skipped.add(src.toString());
                    log.warn("Skip (transform failed): {} -> {}", src.getFileName(), ex.toString());
                }
            }
            trainCount.put(cls, nTrain);
            valCount.put(cls, nVal);
            log.info("Class '{}' -> train={}, val={}", cls, nTrain, nVal);
        }

        Metadata md = new Metadata();
        md.classes = classDirs.stream().map(p -> p.getFileName().toString()).collect(Collectors.toList());
        md.trainCount = trainCount;
        md.valCount = valCount;
        md.targetSize = targetSize;
        md.grayscaleAppearance = grayscaleAppearance;
        md.skipped = skipped;

        Path meta = outRoot.resolve("metadata.json");
        ObjectMapper om = new ObjectMapper();
        om.registerModule(new JavaTimeModule());
        om.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        om.writerWithDefaultPrettyPrinter().writeValue(meta.toFile(), md);

        if (!skipped.isEmpty()) {
            Path skipLog = outRoot.resolve("skipped_images.txt");
            Files.write(skipLog, skipped);
            log.warn("Skipped {} images. See {}", skipped.size(), skipLog.toAbsolutePath());
        }

        log.info("Prepared dataset at {}", outRoot.toAbsolutePath());
        return new PreparedPaths(outRoot, trainRoot, valRoot, meta);
    }

    private static boolean hasAllowedExt(Path p) {
        String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
        for (String ext : ALLOWED_EXT) if (n.endsWith(ext)) return true;
        return false;
    }

    private static boolean isSupportedImage(Path p, List<String> skipped) {
        if (!hasAllowedExt(p)) return false;
        String ct = null;
        try { ct = Files.probeContentType(p); } catch (Exception ignore) {}
        if (ct != null && !ct.startsWith("image/")) {
            if (skipped != null) skipped.add(p + " [mime=" + ct + "]");
            return false;
        }
        return isReadableImage(p, skipped);
    }

    private static boolean isReadableImage(Path p, List<String> skipped) {
        try {
            BufferedImage bi = ImageIO.read(p.toFile());
            boolean ok = bi != null && bi.getWidth() > 1 && bi.getHeight() > 1;
            if (!ok && skipped != null) skipped.add(p.toString());
            return ok;
        } catch (Exception e) {
            if (skipped != null) skipped.add(p.toString());
            return false;
        }
    }

    private static void transformAndSave(Path src, Path dst, int targetSize, boolean grayscaleAppearance) throws IOException {
        BufferedImage img = ImageIO.read(src.toFile());
        if (img == null) throw new IOException("unreadable image");

        BufferedImage scaled = Scalr.resize(img, Scalr.Method.QUALITY, Scalr.Mode.AUTOMATIC, targetSize, targetSize);

        BufferedImage canvasRgb = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = canvasRgb.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, targetSize, targetSize);
        int x = (targetSize - scaled.getWidth()) / 2;
        int y = (targetSize - scaled.getHeight()) / 2;
        g.drawImage(scaled, x, y, null);
        g.dispose();

        if (grayscaleAppearance) {
            BufferedImage gray = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_BYTE_GRAY);
            ColorConvertOp op = new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null);
            op.filter(canvasRgb, gray);
            BufferedImage backToRgb = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
            Graphics2D g2 = backToRgb.createGraphics();
            g2.drawImage(gray, 0, 0, null);
            g2.dispose();
            canvasRgb = backToRgb;
        }

        Files.createDirectories(dst.getParent());
        ImageIO.write(canvasRgb, "jpg", dst.toFile());
    }
}
