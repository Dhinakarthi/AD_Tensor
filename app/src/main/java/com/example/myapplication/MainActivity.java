package com.example.myapplication;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    private Interpreter recognizer;
    private Interpreter detector;
    private static final int REQUEST_CODE_PICK_IMAGE = 1001;
    private static final int REQUEST_PERMISSION_CODE = 1002;

    int INPUT_WIDTH = 608;   // <- Change to your model's input width
    int INPUT_HEIGHT = 800;   // <- Change to your model's input height
    int MAX_CHAR_LEN = 32;   // <- Max characters your model predicts
    int NUM_CLASSES = 80;    // <- Size of your label set

    // Example alphabet (must match your training)
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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        Button button = findViewById(R.id.button);
        detector = loadModelFile(this, "EasyOCR_EasyOCRDetector.tflite");
        recognizer = loadModelFile(this, "EasyOCR_EasyOCRRecognizer.tflite");

        button.setOnClickListener(v -> checkPermission());
    }

    private void checkPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
        != PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_PERMISSION_CODE);
        }
        else {
            openGallery();
        }
    }

    private void openGallery(){
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, REQUEST_CODE_PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE_PICK_IMAGE && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();

            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);

                int[] shape = detector.getInputTensor(0).shape();
                DataType dtype = detector.getInputTensor(0).dataType();
                Log.d("MODEL_SHAPE", Arrays.toString(shape));
                Log.d("MODEL_TYPE", dtype.toString());

                ByteBuffer input = preprocessImage(bitmap, INPUT_WIDTH, INPUT_HEIGHT);

                // Detector output should match [1, 1, 64, 1000] — same shape recognizer expects
                float[][][][] features = new float[1][304][400][2];


                detector.run(input, features);
                float[][][][] recognizerInput = features;

                int MAX_SEQ_LEN = 32;
                int NUM_CLASSES = 80;
                float[][] recognizerOutput = new float[MAX_SEQ_LEN][NUM_CLASSES];

                recognizer.run(recognizerInput, recognizerOutput);

                String textResult = decodeRecognizerOutput(recognizerOutput, alphabet);
                Log.d("OCR_RESULT", "Recognized Text: " + textResult);
            } catch (Exception e){
                Log.d("Errrod dadada", e.getMessage());
            }

        }
    }

    private Interpreter loadModelFile(Context context, String modelName){
        try {
            AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelName);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            return new Interpreter(modelBuffer);
        }
        catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private ByteBuffer preprocessImage(Bitmap bitmap, int width, int height) {
        // Resize to model input size
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, width, height, true);

        // Allocate buffer: width x height x 3 (RGB) x 4 bytes (float)
        ByteBuffer buffer = ByteBuffer.allocateDirect(1 * width * height * 3 * 4);
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[width * height];
        resized.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int pixel : pixels) {
            // Extract RGB channels from ARGB_8888
            buffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f); // R
            buffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);  // G
            buffer.putFloat((pixel & 0xFF) / 255.0f);
        }

        return buffer;
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

            // Avoid duplicates or CTC blank (if blank = alphabet.length)
            if (maxIdx != lastIndex && maxIdx < alphabet.length) {
                result.append(alphabet[maxIdx]);
                lastIndex = maxIdx;
            }
        }

        return result.toString();
    }


}