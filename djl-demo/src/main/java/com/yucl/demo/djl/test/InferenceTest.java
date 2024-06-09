package com.yucl.demo.djl.test;

import static com.yucl.demo.djl.test.TestHelpers.getResourcePath;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.BiFunction;

import org.junit.Test;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtUtil;
import ai.onnxruntime.TensorInfo;

public class InferenceTest {
    // private static final OrtEnvironment env = OrtEnvironment.getEnvironment();

    public static void main(String[] args) throws Exception {
        OrtEnvironment.ThreadingOptions threadOpts = new OrtEnvironment.ThreadingOptions();
        threadOpts.setGlobalInterOpNumThreads(2);
        threadOpts.setGlobalIntraOpNumThreads(2);
        threadOpts.setGlobalDenormalAsZero();
        threadOpts.setGlobalSpinControl(true);

        OrtEnvironment env = OrtEnvironment.getEnvironment(
                OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL, "environmentThreadPool", threadOpts);

        var sessionOptions = new OrtSession.SessionOptions();
        // sessionOptions.addDirectML(0);
        try (
                OrtSession session = env.createSession(
                        "D:\\workspaces\\ai_demo\\learn-ai\\learn-ai\\src\\main\\resources\\partial-inputs-test-2.onnx",
                        sessionOptions)) {
            assertNotNull(session);
            assertEquals(3, session.getNumInputs());
            assertEquals(1, session.getNumOutputs());
            // Input and output collections.
            Map<String, OnnxTensor> inputMap = new HashMap<>();
            Set<String> requestedOutputs = new HashSet<>();

            BiFunction<Result, String, Float> unwrapFunc = (r, s) -> {
                try {
                    return ((float[]) r.get(s).get().getValue())[0];
                } catch (OrtException e) {
                    return Float.NaN;
                }
            };

            // Graph has three scalar inputs, a, b, c, and a single output, ab.
            OnnxTensor a = OnnxTensor.createTensor(env, new float[] { 2.0f });
            OnnxTensor b = OnnxTensor.createTensor(env, new float[] { 3.0f });
            OnnxTensor c = OnnxTensor.createTensor(env, new float[] { 5.0f });

            // Request all outputs, supply all inputs
            inputMap.put("a:0", a);
            inputMap.put("b:0", b);
            inputMap.put("c:0", c);
            requestedOutputs.add("ab:0");
            try (Result r = session.run(inputMap, requestedOutputs)) {
                assertEquals(1, r.size());
                float abVal = unwrapFunc.apply(r, "ab:0");
                assertEquals(6.0f, abVal, 1e-10);
            }

            // Don't specify an output, expect all of them returned.
            try (Result r = session.run(inputMap)) {
                assertEquals(1, r.size());
                float abVal = unwrapFunc.apply(r, "ab:0");
                assertEquals(6.0f, abVal, 1e-10);
            }

            inputMap.clear();
            requestedOutputs.clear();

            // Request single output ab, supply required inputs
            inputMap.put("a:0", a);
            inputMap.put("b:0", b);
            requestedOutputs.add("ab:0");
            try (Result r = session.run(inputMap, requestedOutputs)) {
                assertEquals(1, r.size());
                float abVal = unwrapFunc.apply(r, "ab:0");
                assertEquals(6.0f, abVal, 1e-10);
            }
            inputMap.clear();
            requestedOutputs.clear();

        }
    }

}
