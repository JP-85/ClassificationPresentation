package de.djl.classification;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class RGBPreprocessing extends Preprocessing {

    public RGBPreprocessing(int width, int height) {
        super(width, height, 3);
    }

    public RGBPreprocessing(int width, int height, boolean nomalize) {
        super(width, height, 3,  nomalize);
    }

    @Override
    protected float[] processImage(BufferedImage img) throws IOException {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(img.getScaledInstance(width, height, Image.SCALE_SMOOTH), 0, 0, null);
        g.dispose();

        float[] pixels = new float[width * height * 3]; // RGB
        int idx = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = resized.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int gVal = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                if (normalize) {
                    pixels[idx++] = r / 255.0f;
                    pixels[idx++] = gVal / 255.0f;
                    pixels[idx++] = b / 255.0f;
                } else {
                    pixels[idx++] = r;
                    pixels[idx++] = gVal;
                    pixels[idx++] = b;
                }

            }
        }

        return pixels;
    }
}
