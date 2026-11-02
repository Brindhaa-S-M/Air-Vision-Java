package com.airvision;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/** Calls the existing backend.ml functions without changing ml.py. */
@Service
public class PythonMlBridge {
    private final ObjectMapper mapper;

    @Value("${airvision.project-root:..}")
    private String projectRoot;

    @Value("${airvision.python-command:python}")
    private String pythonCommand;

    public PythonMlBridge(ObjectMapper mapper) {
        this.mapper = mapper;
    }

    private static final String WORKER = """
        import sys, json
        from pathlib import Path
        import pandas as pd
        from backend.ml import load_data, train_model, predict_rows, METRICS_PATH

        request = json.loads(sys.stdin.read() or '{}')
        action = request.get('action')

        if action == 'health':
            result = {'status': 'ok', 'project': 'AirVision', 'model': 'Random Forest Regression'}
        elif action == 'stations':
            frame = load_data()
            records = json.loads(frame.to_json(orient='records'))
            result = {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {'type': 'Point', 'coordinates': [row.get('longitude'), row.get('latitude')]},
                        'properties': {**row, 'is_demo': True}
                    } for row in records
                ]
            }
        elif action == 'metrics':
            if not METRICS_PATH.exists():
                train_model()
            result = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
        elif action == 'train':
            result = train_model()
        elif action == 'predict':
            value = predict_rows([request['payload']])[0]
            result = {
                'predicted_pm25': round(float(value), 2),
                'unit': 'µg/m³',
                'model': 'Random Forest Regression',
                'notice': 'Prototype estimate from synthetic demo training data; not for health or regulatory decisions.'
            }
        elif action == 'predict_rows':
            values = predict_rows(request['rows'])
            result = [float(value) for value in values]
        else:
            raise ValueError('Unknown Python ML action: ' + str(action))

        print(json.dumps(result, allow_nan=False))
        """;

    public Object call(String action) {
        return call(action, Map.of());
    }

    public Object call(String action, Map<String, Object> data) {
        try {
            Path root = Path.of(projectRoot).toAbsolutePath().normalize();
            Process process = new ProcessBuilder(pythonCommand, "-c", WORKER)
                    .directory(root.toFile())
                    .redirectErrorStream(true)
                    .start();

            Map<String, Object> request = new java.util.HashMap<>(data);
            request.put("action", action);
            try (OutputStream input = process.getOutputStream()) {
                input.write(mapper.writeValueAsBytes(request));
            }

            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line);
                }
            }

            boolean finished = process.waitFor(60, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                throw new IllegalStateException("Python ML operation timed out.");
            }
            if (process.exitValue() != 0) {
                throw new IllegalStateException(output.toString());
            }
            return mapper.readValue(output.toString(), Object.class);
        } catch (Exception ex) {
            throw new IllegalStateException("Could not run the existing Python ML module. " +
                    "Check Python, requirements.txt, and the project-root setting. Details: " + ex.getMessage(), ex);
        }
    }

    @SuppressWarnings("unchecked")
    public List<Double> predictRows(List<Map<String, Object>> rows) {
        Object result = call("predict_rows", Map.of("rows", rows));
        return mapper.convertValue(result, new TypeReference<List<Double>>() {});
    }
}
