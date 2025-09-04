package de.djl.classification;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class ImageUtils {

    public static Path saveFeatureGrid(NDArray feature, Path outFile, int tileSize) throws IOException {
        NDArray f = feature.squeeze();
        long[] sh = f.getShape().getShape();
        if (sh.length == 4) {
            if (sh[0] <= 0) throw new IOException("Empty batch in activation");
            f = f.get(new NDIndex("0, :, :, :")).squeeze();
            sh = f.getShape().getShape();
        }
        if (sh.length != 3) throw new IOException("Expected (C,H,W), got " + f.getShape());
        int C = (int) sh[0], H = (int) sh[1], W = (int) sh[2];

        int cols = (int) Math.ceil(Math.sqrt(C));
        int rows = (int) Math.ceil(C / (double) cols);

        BufferedImage grid = new BufferedImage(cols * tileSize, rows * tileSize, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = grid.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, grid.getWidth(), grid.getHeight());

        for (int c = 0; c < C; c++) {
            NDArray ch = f.get(new NDIndex(c + ", :, :"));
            float min = ch.min().getFloat();
            float max = ch.max().getFloat();
            float[] arr = ch.toFloatArray();

            BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_BYTE_GRAY);
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    float v = arr[y * W + x];
                    int gray = (max > min) ? (int) (255f * (v - min) / (max - min)) : 0;
                    int rgb = (gray << 16) | (gray << 8) | gray;
                    img.setRGB(x, y, rgb);
                }
            }

            Image scaled = img.getScaledInstance(tileSize, tileSize, Image.SCALE_SMOOTH);
            BufferedImage tile = new BufferedImage(tileSize, tileSize, BufferedImage.TYPE_BYTE_GRAY);
            Graphics2D tg = tile.createGraphics();
            tg.drawImage(scaled, 0, 0, null);
            tg.dispose();

            int r = c / cols, col = c % cols;
            g.drawImage(tile, col * tileSize, r * tileSize, null);
        }

        g.dispose();
        Files.createDirectories(outFile.getParent());
        ImageIO.write(grid, "png", outFile.toFile());
        return outFile;
    }

    public static Path saveConfusionMatrix2x2(int[][] cm, String[] labels, Path outFile) throws IOException {
        int cell = 100; int pad = 60; int size = cell * 2 + pad * 2; // 2x2 matrix
        BufferedImage img = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.WHITE); g.fillRect(0, 0, size, size);

        g.setColor(Color.GRAY);
        for (int i=0;i<=2;i++) {
            g.drawLine(pad, pad + i*cell, pad + 2*cell, pad + i*cell);
            g.drawLine(pad + i*cell, pad, pad + i*cell, pad + 2*cell);
        }

        g.setColor(Color.BLACK);
        g.setFont(g.getFont().deriveFont(Font.BOLD, 20f));
        for (int r=0; r<2; r++) {
            for (int c=0; c<2; c++) {
                String txt = String.valueOf(cm[r][c]);
                int x = pad + c*cell + cell/2; int y = pad + r*cell + cell/2;
                drawCentered(g, txt, x, y);
            }
        }

        g.setFont(g.getFont().deriveFont(Font.PLAIN, 16f));
        drawCentered(g, "Pred", pad + cell, pad - 30);
        drawCentered(g, "True", pad - 35, pad + cell);
        drawCentered(g, labels[0], pad + cell/2, pad + 2*cell + 20);
        drawCentered(g, labels[1], pad + 3*cell/2, pad + 2*cell + 20);
        drawCentered(g, labels[0], pad - 20, pad + cell/2);
        drawCentered(g, labels[1], pad - 20, pad + 3*cell/2);

        g.dispose();
        Files.createDirectories(outFile.getParent());
        ImageIO.write(img, "png", outFile.toFile());
        return outFile;
    }

    public static Path saveVectorStripe(NDArray vec, Path outFile, int height) throws IOException {
        NDArray v = vec.squeeze();
        long[] sh = v.getShape().getShape();
        int W;
        if (sh.length == 1) {
            W = (int) sh[0];
        } else if (sh.length == 2) {
            long n0 = sh[0];
            long n1 = sh[1];
            W = (int) Math.max(n0, n1);
            v = v.reshape(new ai.djl.ndarray.types.Shape(n0 * n1));
        } else {
            throw new IOException("saveVectorStripe expects 1D/2D, got " + v.getShape());
        }

        float min = v.min().getFloat();
        float max = v.max().getFloat();
        float[] arr = v.toFloatArray();

        BufferedImage img = new BufferedImage(W, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int x = 0; x < W; x++) {
            int gray;
            if (max > min) {
                gray = (int) (255f * (arr[x] - min) / (max - min));
            } else {
                gray = 0;
            }
            int rgb = (gray << 16) | (gray << 8) | gray;
            for (int y = 0; y < height; y++) {
                img.setRGB(x, y, rgb);
            }
        }
        Files.createDirectories(outFile.getParent());
        ImageIO.write(img, "png", outFile.toFile());
        return outFile;
    }


    private static void drawCentered(Graphics2D g, String s, int cx, int cy) {
        var fm = g.getFontMetrics();
        int x = cx - fm.stringWidth(s)/2;
        int y = cy + (fm.getAscent() - fm.getDescent())/2;
        g.drawString(s, x, y);
    }
}
