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

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class MainActivity2 extends AppCompatActivity {

    private Interpreter recognizer;
    private Interpreter detector;
    private ProgressDialog progressDialog;

    int MAX_CHAR_LEN = 32;
    int NUM_CLASSES = 80;

    private final String[] alphabet = {
            "0","1","2","3","4","5","6","7","8","9",
            "A","B","C","D","E","F","G","H","I","J",
            "K","L","M","N","O","P","Q","R","S","T",
            "U","V","W","X","Y","Z",
            "a","b","c","d","e","f","g","h","i","j",
            "k","l","m","n","o","p","q","r","s","t",
            "u","v","w","x","y","z",
            ".",",","-"," ","/","_"
    };

    private final ActivityResultLauncher<Intent> imagePickerLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
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
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_MEDIA_IMAGES}, 1002);
                return;
            }
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
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

    private ByteBuffer preprocessImage(Bitmap bitmap, int width, int height, int channels) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, width, height, true);
        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * width * height * channels);
        buffer.order(ByteOrder.nativeOrder());

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int px = resized.getPixel(x, y);

                // Get channel values from the pixel value.
                int r = Color.red(px);
                int g = Color.green(px);
                int b = Color.blue(px);

                // Normalize channel values to [-1.0, 1.0]. This requirement depends
                // on the model. For example, some models might require values to be
                // normalized to the range [0.0, 1.0] instead.
                float rf = (r - 127) / 255.0f;
                float gf = (g - 127) / 255.0f;
                float bf = (b - 127) / 255.0f;

                buffer.putFloat(rf);
                buffer.putFloat(gf);
                buffer.putFloat(bf);
            }
        }

//        int[] pixels = new int[width * height];
//        resized.getPixels(pixels, 0, width, 0, 0, width, height);
//
//        for (int pixel : pixels) {
//            if (channels == 1) {
//                float r = ((pixel >> 16) & 0xFF) / 255.0f;
//                float g = ((pixel >> 8) & 0xFF) / 255.0f;
//                float b = (pixel & 0xFF) / 255.0f;
//                float gray = (r + g + b) / 3.0f;
//                buffer.putFloat(gray);
//            } else {
//                buffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f); // R
//                buffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);  // G
//                buffer.putFloat((pixel & 0xFF) / 255.0f);         // B
//            }
//        }

        return buffer;
    }

    private void processImage(Bitmap bitmap) {
        progressDialog = ProgressDialog.show(this, "Processing", "Running detection...", true);
        new Thread(() -> {
            try {
                // Get detector input shape
                int[] detectorShape = detector.getInputTensor(0).shape(); // [1, H, W, C]
                int detHeight = detectorShape[1];
                int detWidth = detectorShape[2];
                int detChannels = detectorShape[3];

                Log.v("detHeight", String.valueOf(detHeight));
                Log.v("detWidth", String.valueOf(detWidth));
                Log.v("detChannels", String.valueOf(detChannels));

                ByteBuffer detectorInput = preprocessImage(bitmap, detWidth, detHeight, detChannels);
                float[][][][] detectorOutput = new float[1][detector.getOutputTensor(0).shape()[1]][detector.getOutputTensor(0).shape()[2]][detector.getOutputTensor(0).shape()[3]];
                detector.run(detectorInput, detectorOutput);

                runOnUiThread(() -> {
                    progressDialog.dismiss();
                    showToast("Detected Text: " + "Completed");
                    Log.d("RECOGNIZED_TEXT", detectorInput.toString());
                    Log.d("DETECTOR_OUTPUT", Arrays.deepToString(detectorOutput[0]));
                });


//
//
//
//                // Dummy crop for now
//                int boxWidth = bitmap.getWidth() / 2;
//                int boxHeight = bitmap.getHeight() / 4;
//                int x = (bitmap.getWidth() - boxWidth) / 2;
//                int y = (bitmap.getHeight() - boxHeight) / 2;
//                Rect dummyBox = new Rect(x, y, x + boxWidth, y + boxHeight);
//                Bitmap cropped = safeCrop(bitmap, dummyBox);
//
//                if (cropped == null) {
//                    runOnUiThread(() -> showToast("Invalid bounding box, skipping."));
//                    return;
//                }
//
//                // Get recognizer input shape
//                int[] recShape = recognizer.getInputTensor(0).shape(); // [1, H, W, C]
//                int recHeight = recShape[1];
//                int recWidth = recShape[2];
//                int recChannels = recShape[3];
//
//                ByteBuffer recognizerInput = preprocessImage(cropped, recWidth, recHeight, recChannels);
//                // Get the correct output shape from the model
//                int[] outputShape = recognizer.getOutputTensor(0).shape(); // e.g. [1, 249, 97]
//                int outputTimeSteps = outputShape[1];
//                int outputClasses = outputShape[2];
//
//                // Create array dynamically
//                float[][][] recognizerOutput = new float[1][outputTimeSteps][outputClasses];
//                recognizer.run(recognizerInput, recognizerOutput);
//
//                Log.d("RECOGNIZER_OUTPUT", Arrays.deepToString(recognizerOutput));

                // Decode
//                String recognizedText = decodeRecognizerOutput(recognizerOutput[0], alphabet);



            } catch (Exception e) {
                Log.e("PROCESS_ERROR", e.toString());
                runOnUiThread(() -> {
                    progressDialog.dismiss();
                    showToast("Error: " + e.getMessage());
                });
            }
        }).start();
    }

    private Bitmap safeCrop(Bitmap bitmap, Rect box) {
        int x = Math.max(0, box.left);
        int y = Math.max(0, box.top);
        int right = Math.min(box.right, bitmap.getWidth());
        int bottom = Math.min(box.bottom, bitmap.getHeight());

        int width = right - x;
        int height = bottom - y;

        if (width <= 0 || height <= 0) {
            Log.e("CROP_ERROR", "Invalid crop region: width=" + width + ", height=" + height);
            return null;
        }

        return Bitmap.createBitmap(bitmap, x, y, width, height);
    }

    private String decodeRecognizerOutput(float[][] output, String[] alphabet) {
        StringBuilder result = new StringBuilder();
        int lastIndex = -1;
        for (float[] timestep : output) {
            int maxIdx = 0;
            float maxVal = timestep[0];
            for (int i = 1; i < timestep.length; i++) {
                if (timestep[i] > maxVal) {
                    maxVal = timestep[i];
                    maxIdx = i;
                }
            }
            if (maxIdx != lastIndex && maxIdx < alphabet.length) {
                result.append(alphabet[maxIdx]);
                lastIndex = maxIdx;
            }
        }
        return result.toString();
    }


    private void showToast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
    }
}
