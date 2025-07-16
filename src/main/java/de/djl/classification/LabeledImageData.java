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
}
