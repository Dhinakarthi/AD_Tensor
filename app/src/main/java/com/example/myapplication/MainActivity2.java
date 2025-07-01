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
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
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

    public static class Box {
        public int xMin, yMin, xMax, yMax;

        public Box(int xMin, int yMin, int xMax, int yMax) {
            this.xMin = xMin;
            this.yMin = yMin;
            this.xMax = xMax;
            this.yMax = yMax;
        }

        @Override
        public String toString() {
            return "(" + xMin + "," + yMin + "," + xMax + "," + yMax + ")";
        }
    }

    private void loadLabels(Context context, String fileName) {
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

    private void processImage(Bitmap bitmap) {
        progressDialog = ProgressDialog.show(this, "Processing", "Running detection...", true);
        new Thread(() -> {
            try {
                // 1. Get model input metadata
                int[] detectorShape = detector.getInputTensor(0).shape();  // [1,H,W,C] or [1,C,H,W]
                DataType inputDtype = detector.getInputTensor(0).dataType();
                Tensor.QuantizationParams inputQuant = detector.getInputTensor(0).quantizationParams();

                // 2. Preprocess input image
                Result detectorInput = preprocessDetectorImage(bitmap, detectorShape, inputDtype, inputQuant);

                int[] outputShape = detector.getOutputTensor(0).shape();
                float[][][][] detectorOutput = new float[outputShape[0]][outputShape[1]][outputShape[2]][outputShape[3]];

                detector.run(detectorInput.buffer, detectorOutput);

                float textThreshold = 0.7f;
                float linkThreshold = 0.4f;
                byte[][] mask = postprocessScoreLink(detectorOutput, textThreshold, linkThreshold);
                List<Box> boxes = findConnectedBoxes(mask, 10);

                loadLabels(this, "labels.txt");

                // Recognizer
                int[] recShape = recognizer.getInputTensor(0).shape();  // [1,H,W,C] or [1,C,H,W]
                DataType recDtype = recognizer.getInputTensor(0).dataType();
                Tensor.QuantizationParams recQuant = recognizer.getInputTensor(0).quantizationParams();



                runOnUiThread(() -> {
                    Log.d("Input shape", Arrays.toString(detectorShape));
                    Log.d("Buffer size", String.valueOf(detectorInput.buffer.capacity()));
                    printByteBuffer(detectorInput.buffer, inputDtype, 20);
                    Log.d("RAW_MAPS", Arrays.deepToString(detectorOutput));
                    Log.d("MASK_DEBUG", "Detector mask shape: (" + mask.length + "," + mask[0].length + "), resized image: (" + detectorInput.targetW + "," + detectorInput.targetH + "), original: (" + bitmap.getWidth() + "," + bitmap.getHeight() + ")");
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
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
    }
}
