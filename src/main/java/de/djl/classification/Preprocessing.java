package de.djl.classification;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

public abstract class Preprocessing {
    private static final Logger logger = LoggerFactory.getLogger(Preprocessing.class);

    protected final int width;
    protected final int height;

    public Preprocessing(int width, int height) {
        this.width = width;
        this.height = height;
    }

    /**
     * Muss von Subklassen implementiert werden – verarbeitet ein Bild zu einem float[]-Feature-Vektor.
     */
    protected abstract float[] processImage(BufferedImage img) throws IOException;

    /**
     * Führt das Preprocessing aus und gibt ein CNNDataset zurück.
     *
     * @param datasetName Name des erzeugten Datasets
     * @param rootPath    Root-Verzeichnis mit Subfoldern je Klasse
     * @return CNNDataset mit allen erfolgreich verarbeiteten Bildern
     */
    public CNNDataset run(String datasetName, String rootPath) {
        CNNDataset dataset = new CNNDataset(datasetName);
        Path rootDir = Paths.get(rootPath);

        Map<String, Integer> labelMap = new HashMap<>();
        int[] labelCounter = {0};

        try (Stream<Path> paths = Files.walk(rootDir)) {
            paths.filter(Files::isRegularFile)
                    .filter(path -> {
                        String fileName = path.getFileName().toString().toLowerCase();
                        return fileName.endsWith(".jpg") || fileName.endsWith(".jpeg") || fileName.endsWith(".png");
                    })
                    .forEach(path -> {
                        try {
                            Path relative = rootDir.relativize(path);
                            if (relative.getNameCount() < 1) {
                                logger.warn("Bild liegt nicht in Unterverzeichnis: {}", path);
                                return;
                            }

                            String category = relative.getName(0).toString();
                            int label = labelMap.computeIfAbsent(category, k -> labelCounter[0]++);

                            BufferedImage img = ImageIO.read(path.toFile());
                            if (img == null) {
                                logger.warn("Bild konnte nicht geladen werden (null): {}", path);
                                return;
                            }

                            float[] pixels = processImage(img);
                            if (pixels == null) {
                                logger.warn("Verarbeitung ergab null für Bild: {}", path);
                                return;
                            }

                            dataset.addData(pixels, label, category);
                        } catch (IOException | RuntimeException e) {
                            logger.warn("Fehler bei Bildverarbeitung: {} – {}", path, e.getMessage());
                        }
                    });
        } catch (IOException e) {
            logger.error("Fehler beim Durchlauf des Root-Verzeichnisses: {}", e.getMessage());
        }

        return dataset;
    }
}
