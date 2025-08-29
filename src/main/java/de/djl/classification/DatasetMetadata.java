package de.djl.classification;

import java.util.List;
import java.util.Map;

public class DatasetMetadata {

    private String datasetName;
    private long size;
    private int width;
    private int height;
    private int channels;
    private List<String> synset;
    private Map<String, Long> labelDistribution;

    // Getter/Setter
    public String getDatasetName() { return datasetName; }
    public void setDatasetName(String datasetName) { this.datasetName = datasetName; }

    public long getSize() { return size; }
    public void setSize(long size) { this.size = size; }

    public int getWidth() { return width; }
    public void setWidth(int width) { this.width = width; }

    public int getHeight() { return height; }
    public void setHeight(int height) { this.height = height; }

    public int getChannels() { return channels; }
    public void setChannels(int channels) { this.channels = channels; }

    public List<String> getSynset() { return synset; }
    public void setSynset(List<String> synset) { this.synset = synset; }

    public Map<String, Long> getLabelDistribution() { return labelDistribution; }
    public void setLabelDistribution(Map<String, Long> labelDistribution) { this.labelDistribution = labelDistribution; }

    @Override
    public String toString() {
        return "DatasetMetadata{" +
                "size=" + size +
                ", width=" + width +
                ", height=" + height +
                ", channels=" + channels +
                ", synset=" + synset +
                ", labelDistribution=" + labelDistribution +
                '}';
    }
}
