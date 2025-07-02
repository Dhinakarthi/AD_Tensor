package com.example.myapplication;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;


public class MainActivity2 extends AppCompatActivity {

    private Interpreter recognizer;
    private Interpreter detector;
    private ProgressDialog progressDialog;
    private List<String> labels = null;
    private int blankIndex = -1;

    private final ActivityResultLauncher<Intent> imagePickerLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
        if (result.getResultCode() == RESULT_OK && result.getData() != null) {
            Uri imageUri = result.getData().getData();
            try {
                InputStream inputStream = getContentResolver().openInputStream(imageUri);
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                processImage(bitmap);
            } catch (FileNotFoundException e) {
                showToast("Image not found");
                Log.e("BITMAP_LOAD_ERROR", e.getMessage());
            }
        }
    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main2);
        Button button = findViewById(R.id.button);

        detector = loadModelFile(this, "EasyOCR_EasyOCRDetector.tflite");
        recognizer = loadModelFile(this, "EasyOCR_EasyOCRRecognizer.tflite");

        if (detector == null || recognizer == null) {
            showToast("Failed to load models.");
            return;
        }

        button.setOnClickListener(v -> checkPermission());
    }

    private void checkPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_MEDIA_IMAGES}, 1002);
                return;
            }
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1002);
                return;
            }
        }

        openGallery();
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        imagePickerLauncher.launch(intent);
    }

    private Interpreter loadModelFile(Context context, String modelName) {
        try {
            AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelName);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            return new Interpreter(modelBuffer);
        } catch (Exception e) {
            Log.e("MODEL_LOAD_ERROR", e.getMessage());
            return null;
        }
    }


    public static class Result {
        public ByteBuffer buffer;
        public int targetW;
        public int targetH;
    }

    public static Result preprocessDetectorImage(Bitmap origImg, int[] inputShape, DataType inputDtype, Tensor.QuantizationParams inputQuant) throws Exception {

        if (inputShape.length != 4) {
            throw new IllegalArgumentException("Error: Detector input shape not 4D: " + Arrays.toString(inputShape));
        }

        int batch = inputShape[0];
        int d1 = inputShape[1];
        int d2 = inputShape[2];
        int d3 = inputShape[3];

        String layout;
        int targetH, targetW, channels;

        if (d3 == 1 || d3 == 3) {
            layout = "NHWC";
            targetH = d1;
            targetW = d2;
            channels = d3;
        } else if (d1 == 1 || d1 == 3) {
            layout = "NCHW";
            targetH = d2;
            targetW = d3;
            channels = d1;
        } else {
            throw new IllegalArgumentException("Cannot infer detector layout from shape " + Arrays.toString(inputShape));
        }

        // Resize the image
        Bitmap resized = Bitmap.createScaledBitmap(origImg, targetW, targetH, true);

        int[] pixels = new int[targetW * targetH];
        resized.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH);

        float scale = inputQuant.getScale();
        int zeroPoint = inputQuant.getZeroPoint();

        int bytesPerChannel = getNumBytesPerChannel(inputDtype);
        int numElements = batch * channels * targetH * targetW;
        ByteBuffer buffer = ByteBuffer.allocateDirect(numElements * bytesPerChannel);
        buffer.order(ByteOrder.nativeOrder());

        if (layout.equals("NHWC")) {
            for (int y = 0; y < targetH; y++) {
                for (int x = 0; x < targetW; x++) {
                    int pixel = pixels[y * targetW + x];
                    float r = Color.red(pixel) / 255.0f;
                    float g = Color.green(pixel) / 255.0f;
                    float b = Color.blue(pixel) / 255.0f;

                    if (channels == 1) {
                        float gray = 0.2989f * r + 0.5870f * g + 0.1140f * b;
                        writeToBuffer(buffer, gray, inputDtype, scale, zeroPoint);
                    } else {
                        writeToBuffer(buffer, r, inputDtype, scale, zeroPoint);
                        writeToBuffer(buffer, g, inputDtype, scale, zeroPoint);
                        writeToBuffer(buffer, b, inputDtype, scale, zeroPoint);
                    }
                }
            }
        } else if (layout.equals("NCHW")) {
            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < targetH; y++) {
                    for (int x = 0; x < targetW; x++) {
                        int pixel = pixels[y * targetW + x];
                        float r = Color.red(pixel) / 255.0f;
                        float g = Color.green(pixel) / 255.0f;
                        float b = Color.blue(pixel) / 255.0f;

                        if (channels == 1) {
                            float gray = 0.2989f * r + 0.5870f * g + 0.1140f * b;
                            writeToBuffer(buffer, gray, inputDtype, scale, zeroPoint);
                        } else {
                            if (c == 0) writeToBuffer(buffer, r, inputDtype, scale, zeroPoint);
                            else if (c == 1) writeToBuffer(buffer, g, inputDtype, scale, zeroPoint);
                            else if (c == 2) writeToBuffer(buffer, b, inputDtype, scale, zeroPoint);
                        }
                    }
                }
            }
        }

        buffer.rewind();

        int expectedSize = batch * channels * targetH * targetW * bytesPerChannel;
        if (buffer.capacity() != expectedSize) {
            throw new Exception("Error: Detector preprocessed buffer size " + buffer.capacity() + " != expected " + expectedSize);
        }

        Result result = new Result();
        result.buffer = buffer;
        result.targetW = targetW;
        result.targetH = targetH;
        return result;
    }

    private static void writeToBuffer(ByteBuffer buffer, float value, DataType dtype, float scale, int zeroPoint) {
        if (dtype == DataType.FLOAT32) {
            buffer.putFloat(value);
        } else if (dtype == DataType.UINT8) {
            int quantized = Math.round(value / scale) + zeroPoint;
            quantized = Math.max(0, Math.min(255, quantized));
            buffer.put((byte) (quantized & 0xFF));
        } else if (dtype == DataType.INT8) {
            int quantized = Math.round(value / scale) + zeroPoint;
            quantized = Math.max(-128, Math.min(127, quantized));
            buffer.put((byte) quantized);
        } else {
            throw new IllegalArgumentException("Unsupported input data type: " + dtype);
        }
    }

    private ByteBuffer preprocessRecognizerImage(Bitmap bitmap, int[] inputShape, DataType inputDtype, float scale, int zeroPoint, Integer overrideHeight, Integer overrideWidth, int widthDivisor) {
        if (inputShape.length != 4) {
            throw new IllegalArgumentException("Input shape must be 4D. Got: " + Arrays.toString(inputShape));
        }

        int batch = inputShape[0];
        int d1 = inputShape[1];
        int d2 = inputShape[2];
        int d3 = inputShape[3];

        String layout;
        int channels, targetH, targetW;

        if (d3 == 1 || d3 == 3) {
            layout = "NHWC";
            channels = d3;
            targetH = d1;
            targetW = d2;
        } else if (d1 == 1 || d1 == 3) {
            layout = "NCHW";
            channels = d1;
            targetH = d2;
            targetW = d3;
        } else {
            layout = "NHWC";
            channels = 3;
            targetH = d1;
            targetW = d2;
        }

        if (targetH <= 0) {
            if (overrideHeight == null) {
                throw new IllegalArgumentException("Dynamic height requires overrideHeight.");
            }
            targetH = overrideHeight;
        }

        if (targetW <= 0) {
            int origW = bitmap.getWidth();
            int origH = bitmap.getHeight();
            int newW = (int) Math.ceil(origW * (targetH / (float) origH));
            if (widthDivisor > 1) {
                newW = (int) Math.ceil(newW / (float) widthDivisor) * widthDivisor;
            }
            targetW = newW;
        }

        Bitmap resized = Bitmap.createScaledBitmap(bitmap, targetW, targetH, true);
        int pixelCount = targetH * targetW;
        int[] pixels = new int[pixelCount];
        resized.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH);

        int elementSize = (inputDtype == DataType.FLOAT32) ? 4 : 1;
        int bufferSize = batch * targetH * targetW * channels * elementSize;
        ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize);
        buffer.order(ByteOrder.nativeOrder());

        if (layout.equals("NHWC")) {
            for (int i = 0; i < targetH; i++) {
                for (int j = 0; j < targetW; j++) {
                    int color = pixels[i * targetW + j];
                    int r = (color >> 16) & 0xFF;
                    int g = (color >> 8) & 0xFF;
                    int b = color & 0xFF;

                    if (channels == 1) {
                        float gray = (r + g + b) / 3f / 255f;
                        writeToBuffer(buffer, gray, inputDtype, scale, zeroPoint);
                    } else {
                        writeToBuffer(buffer, r / 255f, inputDtype, scale, zeroPoint);
                        writeToBuffer(buffer, g / 255f, inputDtype, scale, zeroPoint);
                        writeToBuffer(buffer, b / 255f, inputDtype, scale, zeroPoint);
                    }
                }
            }
        } else if (layout.equals("NCHW")) {
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < targetH; i++) {
                    for (int j = 0; j < targetW; j++) {
                        int color = pixels[i * targetW + j];
                        int r = (color >> 16) & 0xFF;
                        int g = (color >> 8) & 0xFF;
                        int b = color & 0xFF;
                        float value;

                        if (channels == 1) {
                            value = (r + g + b) / 3f / 255f;
                        } else {
                            value = ((c == 0) ? r : (c == 1) ? g : b) / 255f;
                        }

                        writeToBuffer(buffer, value, inputDtype, scale, zeroPoint);
                    }
                }
            }
        }

        buffer.rewind();
        return buffer;
    }


    private static int getNumBytesPerChannel(DataType dtype) {
        switch (dtype) {
            case FLOAT32:
                return 4;
            case UINT8:
            case INT8:
                return 1;
            default:
                throw new IllegalArgumentException("Unsupported input data type: " + dtype);
        }
    }

    private List<String> loadLabels(Context context, String fileName) {
        labels = new ArrayList<>();

        try {
            InputStream is = context.getAssets().open(fileName);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            String line;

            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }

            reader.close();

            // Set blankIndex: if not manually set, use last index
            blankIndex = labels.size() - 1;

            Log.d("LABELS_LOADED", "Loaded " + labels.size() + " labels. blankIndex = " + blankIndex);

        } catch (IOException e) {
            Log.e("LABELS_ERROR", "Failed to load labels: " + e.getMessage());
            labels = null;
            blankIndex = -1;
        }

        return labels;
    }


    public static List<Box> findConnectedBoxes(byte[][] mask, int minArea) {
        int h = mask.length;
        int w = mask[0].length;
        boolean[][] visited = new boolean[h][w];
        List<Box> boxes = new ArrayList<>();

        int[][] neighbors = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (mask[y][x] != 0 && !visited[y][x]) {
                    Queue<int[]> queue = new ArrayDeque<>();
                    queue.add(new int[]{y, x});
                    visited[y][x] = true;

                    List<Integer> xs = new ArrayList<>();
                    List<Integer> ys = new ArrayList<>();
                    xs.add(x);
                    ys.add(y);

                    while (!queue.isEmpty()) {
                        int[] pos = queue.poll();
                        int cy = pos[0];
                        int cx = pos[1];

                        for (int[] offset : neighbors) {
                            int ny = cy + offset[0];
                            int nx = cx + offset[1];

                            if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                                if (mask[ny][nx] != 0 && !visited[ny][nx]) {
                                    visited[ny][nx] = true;
                                    queue.add(new int[]{ny, nx});
                                    xs.add(nx);
                                    ys.add(ny);
                                }
                            }
                        }
                    }

                    int xMin = Collections.min(xs);
                    int xMax = Collections.max(xs);
                    int yMin = Collections.min(ys);
                    int yMax = Collections.max(ys);
                    int area = (xMax - xMin + 1) * (yMax - yMin + 1);

                    if (area >= minArea) {
                        boxes.add(new Box(xMin, yMin, xMax, yMax));
                    }
                }
            }
        }

        return boxes;
    }

    private byte[][] postprocessScoreLink(float[][][][] rawMaps, float textThreshold, float linkThreshold) {
        int h = rawMaps[0].length;
        int w = rawMaps[0][0].length;

        byte[][] mask = new byte[h][w];

        int ones = 0;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float score = rawMaps[0][y][x][0]; // text score
                float link = rawMaps[0][y][x][1];  // link score

                // Always apply sigmoid (like in Python)
                score = sigmoid(score);
                link = sigmoid(link);

                byte textMask = (score > textThreshold) ? (byte) 1 : (byte) 0;
                byte linkMask = (link > linkThreshold) ? (byte) 1 : (byte) 0;

                byte combined = (byte) (textMask | linkMask);
                mask[y][x] = combined;

                if (combined == 1) ones++;
            }
        }

        return mask;
    }

    private float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    public String runRecognizer(Interpreter interpreter,
                                Bitmap cropImg,
                                int[] recInputShape,
                                DataType recInputDtype,
                                Tensor.QuantizationParams recInputQuant,
                                Integer overrideHeight,
                                Integer overrideWidth,
                                int widthDivisor,
                                List<String> labels,
                                Integer blankIndex) {

        float scale = recInputQuant != null ? recInputQuant.getScale() : 1.0f;
        int zeroPoint = recInputQuant != null ? recInputQuant.getZeroPoint() : 0;

        ByteBuffer inputBuffer = preprocessRecognizerImage(
                cropImg, recInputShape, recInputDtype, scale, zeroPoint,
                overrideHeight, overrideWidth, widthDivisor);

        printByteBuffer(inputBuffer, recInputDtype, 20);

        // Run inference
        Tensor outputTensor = interpreter.getOutputTensor(0);
        int[] outputShape = outputTensor.shape();
        DataType outputDtype = outputTensor.dataType();

        // Create output array based on model output shape
        float[][][] output = new float[outputShape[0]][outputShape[1]][outputShape[2]];
        interpreter.run(inputBuffer, output);

        Log.v("Outtt", Arrays.deepToString(output));
        // Optional: Output stats
        float min = Float.MAX_VALUE, max = -Float.MAX_VALUE, sum = 0;
        int count = 0;

        for (float[][] row : output) {
            for (float[] timestep : row) {
                for (float val : timestep) {
                    min = Math.min(min, val);
                    max = Math.max(max, val);
                    sum += val;
                    count++;
                }
            }
        }

        float mean = count > 0 ? sum / count : 0;
        Log.d("RECOGNIZER_STATS", String.format("Output shape: %s, min=%.6f, max=%.6f, mean=%.6f",
                Arrays.toString(outputShape), min, max, mean));

        // Decode using greedy CTC
        if (blankIndex == null) {
            blankIndex = labels.size() - 1;
        }

        String text = decodeCTCGreedy(output, labels, blankIndex);
        Log.d("RECOGNIZER_TEXT", "Decoded text: '" + text + "'");

        return text;
    }

    private String decodeCTCGreedy(float[][][] logits, List<String> labels, int blankIndex) {
        if (logits.length != 1) return "?"; // Only batch size 1 supported

        float[][] timeSteps = logits[0];
        StringBuilder decoded = new StringBuilder();
        int prev = -1;

        for (float[] timestep : timeSteps) {
            int maxIdx = argMax(timestep);

            if (maxIdx == prev || maxIdx == blankIndex) {
                prev = maxIdx;
                continue;
            }

            if (maxIdx >= 0 && maxIdx < labels.size()) {
                decoded.append(labels.get(maxIdx));
            } else {
                decoded.append('?');
            }

            prev = maxIdx;
        }

        return decoded.toString();
    }

    private int argMax(float[] array) {
        int maxIdx = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }


    private void processImage(Bitmap bitmap) {
        progressDialog = ProgressDialog.show(this, "Processing", "Running detection...", true);

        new Thread(() -> {
            try {
                // 1. Detector model metadata
                int[] detectorShape = detector.getInputTensor(0).shape();  // [1,H,W,C] or [1,C,H,W]
                DataType inputDtype = detector.getInputTensor(0).dataType();
                Tensor.QuantizationParams inputQuant = detector.getInputTensor(0).quantizationParams();

                // 2. Preprocess image
                Result detectorInput = preprocessDetectorImage(bitmap, detectorShape, inputDtype, inputQuant);

                int[] outputShape = detector.getOutputTensor(0).shape();
                float[][][][] detectorOutput = new float[outputShape[0]][outputShape[1]][outputShape[2]][outputShape[3]];

                detector.run(detectorInput.buffer, detectorOutput);

                float textThreshold = 0.7f;
                float linkThreshold = 0.4f;
                byte[][] mask = postprocessScoreLink(detectorOutput, textThreshold, linkThreshold);
                List<Box> boxes = findConnectedBoxes(mask, 10);

                // Load labels for recognizer
                List<String> labels = loadLabels(this, "labels.txt");

                if (labels == null || labels.isEmpty()) {
                    throw new RuntimeException("Label list is empty or failed to load.");
                }

                // Recognizer metadata
                int[] recShape = recognizer.getInputTensor(0).shape();
                DataType recDtype = recognizer.getInputTensor(0).dataType();
                Tensor.QuantizationParams recQuant = recognizer.getInputTensor(0).quantizationParams();

                int origW = bitmap.getWidth();
                int origH = bitmap.getHeight();
                int resizedW = detectorInput.targetW;
                int resizedH = detectorInput.targetH;

                int maskW = mask[0].length;
                int maskH = mask.length;

                int overrideHeight = 32;         // or any value matching recognizer model height
                int widthDivisor = 1;            // used for padding width if required
                int blankIndex = labels.size() - 1;

                for (Box box : boxes) {
                    Rect boxOrig = mapBoxMaskToOriginal(box, maskW, maskH, resizedW, resizedH, origW, origH);

                    int x1 = boxOrig.left;
                    int y1 = boxOrig.top;
                    int x2 = boxOrig.right;
                    int y2 = boxOrig.bottom;

                    if ((x2 - x1) < 5 || (y2 - y1) < 5) {
                        Log.w("CROP", "Skipping too-small region: (" + x1 + "," + y1 + "," + x2 + "," + y2 + ")");
                        continue;
                    }

                    if (x2 > x1 && y2 > y1) {
                        Bitmap crop = Bitmap.createBitmap(bitmap, x1, y1, x2 - x1, y2 - y1);
                        Log.v("CROP", "Cropped bitmap size: width=" + crop.getWidth() + ", height=" + crop.getHeight() + " from (" + x1 + "," + y1 + ") to (" + x2 + "," + y2 + ")");

                        String result = runRecognizer(recognizer, crop, recShape, recDtype, recQuant, overrideHeight, null, widthDivisor, labels, blankIndex);

                        Log.v("RESULT", result);

                    } else {
                        Log.w("CROP", "Invalid crop size: (" + x1 + "," + y1 + "," + x2 + "," + y2 + ")");
                    }
                }

                runOnUiThread(() -> {
                    Log.d("Input shape", Arrays.toString(detectorShape));
                    Log.d("Buffer size", String.valueOf(detectorInput.buffer.capacity()));
                    printByteBuffer(detectorInput.buffer, inputDtype, 20);
                    Log.d("RAW_MAPS", Arrays.deepToString(detectorOutput));
                    Log.d("MASK_DEBUG", "Detector mask shape: (" + mask.length + "," + mask[0].length + "), resized image: (" + resizedW + "," + resizedH + "), original: (" + origW + "," + origH + ")");
                    Log.d("BOXES", "Found " + boxes.size() + " regions in mask coords.");
                    for (Box box : boxes) {
                        Log.d("BOX_COORDS", box.toString());
                    }
                    Log.v("recShape", Arrays.toString(recShape));
                    Log.v("recDtype", recDtype.toString());
                    Log.v("recQuant", "scale=" + recQuant.getScale() + ", zeroPoint=" + recQuant.getZeroPoint());
                    progressDialog.dismiss();
                    showToast("Detection completed");
                });

            } catch (Exception e) {
                Log.e("PROCESS_ERROR", Log.getStackTraceString(e));
                runOnUiThread(() -> {
                    progressDialog.dismiss();
                    showToast("Error: " + e.getMessage());
                });
            }
        }).start();
    }


    public static Rect mapBoxMaskToOriginal(Box box, int maskW, int maskH, int resizedW, int resizedH, int origW, int origH) {
        int xMin = box.x1;
        int yMin = box.y1;
        int xMax = box.x2;
        int yMax = box.y2;

        // Step 1: map mask coords to resized coords
        int x1_r = xMin * 2;
        int x2_r = (xMax + 1) * 2;
        int y1_r = yMin * 2;
        int y2_r = (yMax + 1) * 2;

        // Step 2: resized → original
        float fx = (float) origW / resizedW;
        float fy = (float) origH / resizedH;

        int x1 = Math.round(x1_r * fx);
        int x2 = Math.round(x2_r * fx);
        int y1 = Math.round(y1_r * fy);
        int y2 = Math.round(y2_r * fy);

        // Clip
        x1 = Math.max(0, Math.min(x1, origW - 1));
        x2 = Math.max(1, Math.min(x2, origW));
        y1 = Math.max(0, Math.min(y1, origH - 1));
        y2 = Math.max(1, Math.min(y2, origH));

        if (x2 <= x1) x2 = Math.min(origW, x1 + 1);
        if (y2 <= y1) y2 = Math.min(origH, y1 + 1);

        return new Rect(x1, y1, x2, y2);
    }


    private void printByteBuffer(ByteBuffer buffer, DataType dtype, int maxElements) {
        buffer.rewind(); // Start from beginning
        int count = 0;

        if (dtype == DataType.FLOAT32) {
            while (buffer.remaining() >= 4 && count < maxElements) {
                float val = buffer.getFloat();
                Log.d("BUFFER_FLOAT32", "val = " + val);
                count++;
            }
        } else if (dtype == DataType.UINT8 || dtype == DataType.INT8) {
            while (buffer.remaining() >= 1 && count < maxElements) {
                byte b = buffer.get();
                Log.d("BUFFER_INT8", "val = " + (b & 0xFF)); // Use & 0xFF for unsigned
                count++;
            }
        } else {
            Log.e("BUFFER_PRINT", "Unsupported dtype: " + dtype);
        }

        buffer.rewind(); // Reset after reading
    }


    private void showToast(String message) {
        Log.v("Error", message);
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
    }
}
