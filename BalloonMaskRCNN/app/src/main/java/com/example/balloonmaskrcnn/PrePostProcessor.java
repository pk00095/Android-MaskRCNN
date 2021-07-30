// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package com.example.balloonmaskrcnn;


import android.graphics.Rect;
import android.graphics.RectF;

import java.util.ArrayList;

class Result {
    int classIndex;
    Float score;
    RectF rect;

    public Result(int cls, Float output, RectF rect) {
        this.classIndex = cls;
        this.score = output;
        this.rect = rect;
    }
};

public class PrePostProcessor {
    // for yolov5 model, no need to apply MEAN and STD
    public final static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    // the model expects tensors to be in range 0-255, but converting bitmap to tensor normalizes and puts it
    // in the range of 0 to 1. So below instead of STD being 1 we're making it 1/255 to force it into range
    // 0-255
    public final static float[] NO_STD_RGB = new float[] {(float) (1.0/255.0), (float) (1.0/255.0), (float) (1.0/255.0)};

    // model input image size
    public final static int INPUT_WIDTH = 640;
    public final static int INPUT_HEIGHT = 640;
    public final static int OUTPUT_COLUMN = 6; // left, top, right, bottom, score and label

    static String[] mClasses;

    static ArrayList<Result> outputsToPredictions(int countResult, float[] outputs, float imgScaleX, float imgScaleY, float ivScaleX, float ivScaleY, float startX, float startY) {
        ArrayList<Result> results = new ArrayList<>();
        for (int i = 0; i< countResult; i++) {
            float left = outputs[i* OUTPUT_COLUMN];
            float top = outputs[i* OUTPUT_COLUMN +1];
            float right = outputs[i* OUTPUT_COLUMN +2];
            float bottom = outputs[i* OUTPUT_COLUMN +3];

            left = imgScaleX * left;
            top = imgScaleY * top;
            right = imgScaleX * right;
            bottom = imgScaleY * bottom;

//            Rect rect = new Rect((int)(startX+ivScaleX*left), (int)(startY+top*ivScaleY), (int)(startX+ivScaleX*right), (int)(startY+ivScaleY*bottom));
//            RectF rect = new RectF(startX+ivScaleX*left, startY+top*ivScaleY, startX+ivScaleX*right, startY+ivScaleY*bottom);
            RectF rect = new RectF(left, top, right, bottom);
            Result result = new Result((int)outputs[i* OUTPUT_COLUMN +5], outputs[i* OUTPUT_COLUMN +4], rect);
            results.add(result);
        }
        return results;
    }
}
