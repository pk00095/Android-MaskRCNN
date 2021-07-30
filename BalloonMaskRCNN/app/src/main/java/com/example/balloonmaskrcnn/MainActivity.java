package com.example.balloonmaskrcnn;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffColorFilter;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

import org.pytorch.IValue;
import org.pytorch.Module;
import java.io.IOException;

import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Map;


public class MainActivity extends AppCompatActivity {

    public static final int GALLERY_REQ_CODE = 101;
    public static final double MASK_THRESH = 0.9;

    static {
        if (!NativeLoader.isInitialized()) {
            NativeLoader.init(new SystemDelegate());
        }
        NativeLoader.loadLibrary("pytorch_jni");
        NativeLoader.loadLibrary("torchvision_ops");
    }

    Button galleryBtn, runAlgo;
    ImageView imageViewer;
    Uri contenturi;
    Bitmap mBitmap;
    private Module mModule = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageViewer = findViewById(R.id.imageView);
        galleryBtn = findViewById(R.id.galleryBtn);
        runAlgo = findViewById(R.id.runAlgo);
        mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), "d2goBalloonMrcnn.pt");

        galleryBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent gallery = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(gallery, GALLERY_REQ_CODE);
            }
        });

        runAlgo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                runMmaskRcnn();
            }
        });
    }

    private void runMmaskRcnn() {
        try {
//            InputStream ims = getContentResolver().openInputStream(contenturi);
//            mBitmap = BitmapFactory.decodeStream(ims);
            mBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), contenturi);
            Log.i("inference", "bitmap loaded");
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        mImgScaleX = (float)mBitmap.getWidth() / PrePostProcessor.INPUT_WIDTH;
        mImgScaleY = (float)mBitmap.getHeight() / PrePostProcessor.INPUT_HEIGHT;

        mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float)imageViewer.getWidth() / mBitmap.getWidth() : (float)imageViewer.getHeight() / mBitmap.getHeight());
        mIvScaleY  = (mBitmap.getHeight() > mBitmap.getWidth() ? (float)imageViewer.getHeight() / mBitmap.getHeight() : (float)imageViewer.getWidth() / mBitmap.getWidth());

        mStartX = (imageViewer.getWidth() - mIvScaleX * mBitmap.getWidth())/2;
        mStartY = (imageViewer.getHeight() -  mIvScaleY * mBitmap.getHeight())/2;
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.INPUT_WIDTH, PrePostProcessor.INPUT_HEIGHT, true);

        Log.i("inference", "colour space "+resizedBitmap.getConfig());

        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * resizedBitmap.getWidth() * resizedBitmap.getHeight());
        TensorImageUtils.bitmapToFloatBuffer(resizedBitmap, 0,0,resizedBitmap.getWidth(),resizedBitmap.getHeight(), PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB, floatBuffer, 0);
        final Tensor inputTensor =  Tensor.fromBlob(floatBuffer, new long[] {3, resizedBitmap.getHeight(), resizedBitmap.getWidth()});


        final long startTime = SystemClock.elapsedRealtime();
        IValue[] outputTuple = mModule.forward(IValue.listFrom(inputTensor)).toTuple();
        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.i("inference",  "inference time (ms): " + inferenceTime);

        final Map<String, IValue> map = outputTuple[1].toList()[0].toDictStringKey();
        float[] boxesData = new float[]{};
        float[] scoresData = new float[]{};
        long[] labelsData = new long[]{};
        float[] masksData = new float[]{};
        if (map.containsKey("boxes")) {
            final Tensor boxesTensor = map.get("boxes").toTensor();
            final Tensor scoresTensor = map.get("scores").toTensor();
            final Tensor labelsTensor = map.get("labels").toTensor();
            final Tensor maskTensor = map.get("masks").toTensor();
            long[] maskShape = maskTensor.shape();
            Log.i("inference", "masks :: "+maskTensor);
            boxesData = boxesTensor.getDataAsFloatArray();
            scoresData = scoresTensor.getDataAsFloatArray();
            labelsData = labelsTensor.getDataAsLongArray();
            masksData = maskTensor.getDataAsFloatArray();

            final int maskHeight = (int) maskShape[2];
            final int maskWidth = (int) maskShape[3];
            ArrayList<Bitmap> predMasks = new ArrayList<Bitmap>();
            Log.i("inference", "mask shape :: "+maskShape[0]+","+maskHeight+","+maskWidth);
            Log.i("inference", "masks :: "+masksData.length);
//            Log.i("inference", "labels :: "+labelsData.toString());

            final int n = scoresData.length;
            Log.i("inference", "got "+n+ " preds");
            float[] outputs = new float[n * PrePostProcessor.OUTPUT_COLUMN];

            int count = 0;
            for (int i = 0; i < n; i++) {
                if (scoresData[i] < 0.5)
                    continue;

                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 0] = boxesData[4 * i + 0];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 1] = boxesData[4 * i + 1];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 2] = boxesData[4 * i + 2];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 3] = boxesData[4 * i + 3];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 4] = scoresData[i];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 5] = labelsData[i] - 1;

                Bitmap maskBitmap = Bitmap.createBitmap(maskWidth, maskHeight, Bitmap.Config.ARGB_8888);
                maskBitmap.setHasAlpha(true);
                Canvas canvas = new Canvas(maskBitmap);
                Paint paint = new Paint();

                int num_zero = 0;
                int num_ones = 0;
                for (int j=0; j<maskWidth; j++){
                    for (int k=0; k<maskHeight; k++){
                        if (masksData[(i*maskHeight*maskWidth) + (j*maskWidth) +k] > MASK_THRESH){
                            maskBitmap.setPixel(j,k, Color.argb(100, 0, 255,0));
                            num_ones++;
                        }else {
                            num_zero++;
                        }
                    }
                }

                predMasks.add(maskBitmap);
                count++;
            }

            final ArrayList<Result> results = PrePostProcessor.outputsToPredictions(count, outputs, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);

            for (Result r: results){
                Log.i("inference", String.valueOf(r.rect));
                Log.i("inference", String.valueOf(r.score));

            }
            Bitmap annotatedBitmap = drawBboxes(mBitmap, results, 0.8, predMasks);
            imageViewer.setImageDrawable(new BitmapDrawable(getResources(), annotatedBitmap));
        }
    }

    private Bitmap drawBboxes(Bitmap bitmap, ArrayList<Result> results, double bbox_thresh, ArrayList<Bitmap> masks) {
        Bitmap tempBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), bitmap.getConfig());
        tempBitmap.setHasAlpha(true);
        Canvas canvas = new Canvas(tempBitmap);
        //Draw the image bitmap into the cavas
        canvas.drawBitmap(bitmap, 0, 0, null);

        Paint bbox = new Paint(Paint.ANTI_ALIAS_FLAG);
        bbox.setStyle(Paint.Style.STROKE);
        bbox.setColor(Color.GREEN);
        bbox.setStrokeWidth(3);

        RectF location;
//        String category;
        float score;
        Log.i("annotate", "drawing");
        Log.i("annotate", String.valueOf(results.size()));

        for (int i=0; i<results.size(); i++){
            score = results.get(i).score;
            Log.i("annotate", "index :: "+String.valueOf(i));
            Log.i("annotate", "score :: "+String.valueOf(score));
            if (score > bbox_thresh){

                location = results.get(i).rect;
                Bitmap bmp = Bitmap.createScaledBitmap(masks.get(i), (int) location.width(), (int) location.height(), true);
                canvas.drawRect(location, bbox);
                canvas.drawBitmap(bmp, location.left, location.top, null);
                Log.i("annotate", String.valueOf(score));
                Log.i("annotate", String.valueOf(location));
            }

        }
        Log.i("annotate", "drawing done");

        return tempBitmap;
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == GALLERY_REQ_CODE) {
            if (resultCode == Activity.RESULT_OK) {
                contenturi = data.getData();
                imageViewer.setImageURI(contenturi);
                Log.i("gallery", String.valueOf(contenturi));
            }
        }
    }
    
}