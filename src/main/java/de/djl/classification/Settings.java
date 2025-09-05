package de.djl.classification;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;

public class Settings {

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Setting {
        public String name;
        public int stride;
        public int[] kernel;
        public int[] maxPoolSize;
        public String optimizer;
        public double learningRate;
        public int convLayers;
        public int[] denseUnits;
        public String activation;
        public int batchSize;
        public double dropout;
        public Double leakyAlpha;

        public Integer baseChannels;
        public Integer maxChannels;
        public Boolean globalAvgPool;

        @Override public String toString() {
            return String.format(Locale.ROOT,
                    "%s conv=%d kernel=%s stride=%d pool=%s dense=%s act=%s opt=%s lr=%.4g bs=%d drop=%.2f base=%s max=%s gap=%s",
                    name, convLayers, Arrays.toString(kernel), stride, Arrays.toString(maxPoolSize),
                    Arrays.toString(denseUnits), activation, optimizer, learningRate, batchSize, dropout,
                    baseChannels, maxChannels, globalAvgPool);
        }
    }

    private final Map<String, Setting> byName = new LinkedHashMap<>();

    public Settings(List<Setting> list) {
        for (Setting s : list) byName.put(s.name, s);
    }

    public Setting get(String name) {
        Setting s = byName.get(name);
        if (s == null) throw new IllegalArgumentException("Unknown setting: " + name + " (available: " + byName.keySet() + ")");
        return s;
    }

    public static Settings loadFromResources(String resourcePath) throws IOException {
        ObjectMapper om = new ObjectMapper();
        try (InputStream is = Settings.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (is == null) throw new IOException("settings file not found in resources: " + resourcePath);
            List<Setting> list = om.readValue(is, new TypeReference<>() {
            });
            return new Settings(list);
        }
    }
}
