package de.djl.classification;

import java.io.Serial;
import java.io.Serializable;

public class LabeledImageData implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private final int label;
    private final float[] pixels;

    public LabeledImageData(int label, float[] pixels) {
        this.label = label;
        this.pixels = pixels;
    }

    public int getLabel() {
        return label;
    }

    public float[] getPixels() {
        return pixels;
    }

    /**
     * Gibt das Bild zurück als:
     * - float[height][width] bei 1 Channel
     * - float[channels][height][width] bei >1 Channel
     */
    public Object getImage(int width, int height, int channels) {
        if (channels == 1) {
            float[][] img = new float[height][width];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    img[y][x] = pixels[y * width + x];
                }
            }
            return img; // float[][]
        } else {
            float[][][] img = new float[channels][height][width];
            int idx = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < channels; c++) {
                        img[c][y][x] = pixels[idx++];
                    }
                }
            }
            return img; // float[][][]
        }
    }
}
