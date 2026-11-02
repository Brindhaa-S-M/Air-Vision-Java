package com.airvision;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
public class AirVisionController implements WebMvcConfigurer {
    private final PythonMlBridge ml;

    @org.springframework.beans.factory.annotation.Value("${airvision.project-root:..}")
    private String projectRoot;

    public AirVisionController(PythonMlBridge ml) {
        this.ml = ml;
    }

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        String frontend = Path.of(projectRoot).toAbsolutePath().normalize()
                .resolve("frontend").toUri().toString();
        registry.addResourceHandler("/**").addResourceLocations(frontend);
    }

    @GetMapping("/api/health")
    public Object health() {
        return ml.call("health");
    }

    @GetMapping("/api/stations")
    public Object stations() {
        return ml.call("stations");
    }

    @GetMapping("/api/metrics")
    public Object metrics() {
        return ml.call("metrics");
    }

    @PostMapping("/api/train")
    public Object train() {
        return ml.call("train");
    }

    @PostMapping("/api/predict")
    public Object predict(@RequestBody Map<String, Object> payload) {
        validatePrediction(payload);
        return ml.call("predict", Map.of("payload", payload));
    }

    @GetMapping("/api/prediction-grid")
    public Object predictionGrid(
            @RequestParam(defaultValue = "12.7") double south,
            @RequestParam(defaultValue = "80.0") double west,
            @RequestParam(defaultValue = "13.3") double north,
            @RequestParam(defaultValue = "80.4") double east,
            @RequestParam(defaultValue = "8") int resolution,
            @RequestParam(defaultValue = "20") double no2,
            @RequestParam(defaultValue = "0.3") double aod,
            @RequestParam(defaultValue = "30") double temperature,
            @RequestParam(defaultValue = "65") double humidity,
            @RequestParam(defaultValue = "2") double wind_speed,
            @RequestParam(defaultValue = "1") int month) {

        if (!(south >= -90 && south < north && north <= 90 &&
                west >= -180 && west < east && east <= 180)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Invalid geographic bounds.");
        }
        if (resolution < 3 || resolution > 30) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "resolution must be between 3 and 30.");
        }
        if (no2 < 0 || aod < 0 || humidity < 0 || humidity > 100 || wind_speed < 0 || month < 1 || month > 12) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "One or more feature values are outside valid ranges.");
        }

        List<Map<String, Object>> rows = new ArrayList<>();
        for (int i = 0; i < resolution; i++) {
            double latitude = south + (north - south) * i / (resolution - 1.0);
            for (int j = 0; j < resolution; j++) {
                double longitude = west + (east - west) * j / (resolution - 1.0);
                Map<String, Object> row = new HashMap<>();
                row.put("latitude", latitude);
                row.put("longitude", longitude);
                row.put("no2", no2);
                row.put("aod", aod);
                row.put("temperature", temperature);
                row.put("humidity", humidity);
                row.put("wind_speed", wind_speed);
                row.put("month", month);
                rows.add(row);
            }
        }

        List<Double> predictions = ml.predictRows(rows);
        List<Map<String, Object>> points = new ArrayList<>();
        for (int i = 0; i < rows.size(); i++) {
            Map<String, Object> point = new HashMap<>();
            point.put("latitude", rows.get(i).get("latitude"));
            point.put("longitude", rows.get(i).get("longitude"));
            point.put("pm25", Math.round(predictions.get(i) * 100.0) / 100.0);
            points.add(point);
        }

        return Map.of(
                "model", "Random Forest Regression",
                "unit", "µg/m³",
                "resolution", resolution,
                "points", points,
                "notice", "Spatial model output from synthetic demo training data. It is not validated real-world downscaling."
        );
    }

    private void validatePrediction(Map<String, Object> p) {
        checkRange(p, "latitude", -90, 90);
        checkRange(p, "longitude", -180, 180);
        checkMinimum(p, "no2", 0);
        checkMinimum(p, "aod", 0);
        checkRange(p, "humidity", 0, 100);
        checkMinimum(p, "wind_speed", 0);
        checkRange(p, "month", 1, 12);
        for (String key : List.of("temperature")) {
            if (!(p.get(key) instanceof Number)) {
                throw new ResponseStatusException(HttpStatus.BAD_REQUEST, key + " must be a number.");
            }
        }
    }

    private void checkRange(Map<String, Object> p, String key, double min, double max) {
        if (!(p.get(key) instanceof Number n) || n.doubleValue() < min || n.doubleValue() > max) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, key + " must be between " + min + " and " + max + ".");
        }
    }

    private void checkMinimum(Map<String, Object> p, String key, double min) {
        if (!(p.get(key) instanceof Number n) || n.doubleValue() < min) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, key + " must be at least " + min + ".");
        }
    }
}
