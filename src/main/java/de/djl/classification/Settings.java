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
        public int[] kernel;        // [kh, kw]
        public int[] maxPoolSize;   // [ph, pw]
        public String optimizer;    // adam | sgd | rmsprop
        public double learningRate;
        public int convLayers;      // number of conv blocks
        public int[] denseUnits;    // e.g., [128], [256,128]
        public String activation;   // relu | leakyrelu
        public int batchSize;
        public double dropout;      // 0..1
        public Double leakyAlpha;   // optional

        // NEW: let you tune channel width & head
        public Integer baseChannels;   // default 64 (or 32/16 for small images)
        public Integer maxChannels;    // default 512
        public Boolean globalAvgPool;  // default false

        @Override public String toString() {
            return String.format(Locale.ROOT,
                    "%s conv=%d kernel=%s stride=%d pool=%s dense=%s act=%s opt=%s lr=%.4g bs=%d drop=%.2f base=%s max=%s gap=%s",
                    name, convLayers, Arrays.toString(kernel), stride, Arrays.toString(maxPoolSize),
                    Arrays.toString(denseUnits), activation, optimizer, learningRate, batchSize, dropout,
                    String.valueOf(baseChannels), String.valueOf(maxChannels), String.valueOf(globalAvgPool));
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

    public Set<String> names() { return Collections.unmodifiableSet(byName.keySet()); }

    public static Settings loadFromResources(String resourcePath) throws IOException {
        ObjectMapper om = new ObjectMapper();
        try (InputStream is = Settings.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (is == null) throw new IOException("settings file not found in resources: " + resourcePath);
            List<Setting> list = om.readValue(is, new TypeReference<List<Setting>>(){});
            return new Settings(list);
        }
    }
}
