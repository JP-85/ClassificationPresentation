package de.djl.classification;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class GrayscalePreprocessing extends Preprocessing {

    public GrayscalePreprocessing(int width, int height, boolean normalize) { super(width, height, 1, normalize); }
    public GrayscalePreprocessing(int width, int height) { super(width, height, 1); }

    @Override
    protected float[] processImage(BufferedImage img) throws IOException {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = resized.createGraphics();
        g.drawImage(img.getScaledInstance(width, height, Image.SCALE_SMOOTH), 0, 0, null);
        g.dispose();

        float[] pixels = new float[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = resized.getRaster().getSample(x, y, 0);
                if (normalize) {
                    pixels[y * width + x] = gray / 255.0f; // Normalisiert
                } else {
                    pixels[y * width + x] = gray;
                }

            }
        }

        return pixels;
    }
}
