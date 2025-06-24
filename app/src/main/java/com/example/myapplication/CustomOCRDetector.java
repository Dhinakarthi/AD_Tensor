package com.example.myapplication;

public class CustomOCRDetector {
    private static final String TAG = "CustomOCRDetector";

    // Model configuration - Adjust these based on your model
    private static final int INPUT_WIDTH = 224;
    private static final int INPUT_HEIGHT = 224;
    private static final int INPUT_CHANNELS = 3;
    private static final int OUTPUT_SIZE = 1000; // Adjust based on your model output
    private static final String MODEL_FILE = "custom_ocr_model.tflite";
}
