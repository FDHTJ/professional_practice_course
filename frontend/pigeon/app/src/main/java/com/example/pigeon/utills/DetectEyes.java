package com.example.pigeon.utills;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.example.pigeon.R;
import com.example.pigeon.databinding.ActivityMainBinding;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.pigeon.databinding.ActivityMainBinding;
import android.content.res.AssetManager;
import android.widget.Toast;

import org.json.JSONException;

import java.io.IOException;
import java.io.InputStream;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class DetectEyes {
      // Used to load the 'pigeon' library on application startup.
      static {
         System.loadLibrary("pigeon");
      }
    private static DetectEyes instance;
    private boolean isInitialized = false;

    static {
        System.loadLibrary("pigeon");
    }

    private DetectEyes() {}

    public static DetectEyes getInstance() {
        if (instance == null) {
            instance = new DetectEyes();
        }
        return instance;
    }
    public boolean init(AssetManager assets) {
        if (isInitialized) return true;
        isInitialized = initYolo(assets);
        return isInitialized;
    }
    public native boolean initYolo(AssetManager mgr);
    public native String detect(Bitmap bitmap); // 返回结果字符串


    public String getYoloDetectResult(Bitmap bitmap) {
        try {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            String result = detect(mutableBitmap);
            System.out.println("检测结果: " + result);
            return result;
         } catch (Exception e) {
            e.printStackTrace();
            return Constant.FAIL_TO_DETECT_YOLO;
         }

      }

      public String drawRects(Bitmap bitmap, String result) {
          if (result == null || result.isEmpty()) return Constant.FAIL_TO_DETECT_YOLO;

          Canvas canvas = new Canvas(bitmap);
          Paint paint = new Paint();
          paint.setColor(Color.RED);
          paint.setStyle(Paint.Style.STROKE);
          paint.setStrokeWidth(3);
          paint.setTextSize(20);

          // 结果格式: "label,prob,x,y,w,h|..."
          String[] boxes = result.split("\\|");
          if (boxes.length == 0) {
              return Constant.FAIL_TO_DETECT_YOLO;
          }

          for (String box : boxes) {
              if (box.isEmpty()) continue;
              String[] data = box.split(",");
              if (data.length < 6) continue;

              int x = Integer.parseInt(data[2]);
              int y = Integer.parseInt(data[3]);
              int w = Integer.parseInt(data[4]);
              int h = Integer.parseInt(data[5]);

              // 画框
              canvas.drawRect(x, y, x + w, y + h, paint);
              // 画文字
              canvas.drawText("Eye", x, y - 10, paint);
              return box;
          }
          return Constant.FAIL_TO_DETECT_YOLO;
      }
    public Bitmap getBitmapFromUri(Uri uri, ContentResolver contentResolver) throws IOException {
        // 检查 Android 版本，P (9.0) 以上推荐用 ImageDecoder
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            ImageDecoder.Source source = ImageDecoder.createSource(contentResolver, uri);
            return ImageDecoder.decodeBitmap(source).copy(Bitmap.Config.ARGB_8888, true);
        } else {
            // 老版本手机用这个方法
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri);
            return bitmap.copy(Bitmap.Config.ARGB_8888, true);
        }
    }
    public String uploadImageToServerForComparison(Bitmap bitmap1, Bitmap bitmap2, com.example.pigeon.utills.Callback callback) throws ExecutionException, InterruptedException {
        // 1. 把 Bitmap 转换成 byte[] 数组 (也就是图片文件流)
        ByteArrayOutputStream stream1 = new ByteArrayOutputStream();
        ByteArrayOutputStream stream2 = new ByteArrayOutputStream();
        // 压缩格式 PNG (无损) 或 JPEG
        bitmap1.compress(Bitmap.CompressFormat.PNG, 100, stream1);
        bitmap2.compress(Bitmap.CompressFormat.PNG, 100, stream2);
        byte[] byteArray1 = stream1.toByteArray();
        byte[] byteArray2 = stream2.toByteArray();
        MultipartBody.Builder builder = new MultipartBody.Builder()
                .setType(MultipartBody.FORM);
        String left_eye= UUID.randomUUID().toString();
        String right_eye= UUID.randomUUID().toString();
        // --- 添加第一张图 ---
        builder.addFormDataPart("file_1", left_eye+".png",
                RequestBody.create(byteArray1, MediaType.parse("image/png")));
        System.out.println(left_eye+".png");
        // --- 添加第二张图 ---
        builder.addFormDataPart("file_2", right_eye+".png",
                RequestBody.create(byteArray2, MediaType.parse("image/png")));
        String serverUrl = Constant.REQUEST_URL+"/upload/eye";
        // 4. 发送请求
        Request request = new Request.Builder()
                .url(serverUrl)
                .post(builder.build())
                .build();
        // 2. 准备 OkHttp 客户端
        OkHttpClient client = new OkHttpClient();
        CompletableFuture<String> future = new CompletableFuture<>();


        // 5. 异步发送请求 (不会卡死界面)
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
//                e.printStackTrace();
                callback.onError(e);
            }
            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    // 服务器返回的 JSON
                    String responseData = response.body().string();
                    System.out.println("服务器响应: " + responseData);
                    future.complete(responseData);

                    callback.onSuccess(responseData);

                } else {
                    future.complete(Constant.SERVER_ERROR);
                }
            }
        });
        return future.get();
    }
    public String uploadImageToServerForRetrieval(Bitmap bitmap, com.example.pigeon.utills.Callback callback) throws ExecutionException, InterruptedException {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
        byte[] byteArray = stream.toByteArray();
        MultipartBody.Builder builder = new MultipartBody.Builder()
                .setType(MultipartBody.FORM);
        String file_name=UUID.randomUUID().toString();
        builder.addFormDataPart("file", file_name+".png",
                RequestBody.create(byteArray, MediaType.parse("image/png")));
        String serverUrl = Constant.REQUEST_URL+"/upload/eye_retrieval";
        Request request = new Request.Builder()
                .url(serverUrl)
                .post(builder.build())
                .build();
        OkHttpClient client = new OkHttpClient().newBuilder().callTimeout(15, TimeUnit.SECONDS).build();
        CompletableFuture<String> future = new CompletableFuture<>();


        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
//                e.printStackTrace();
                callback.onError(e);
            }
            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    // 服务器返回的 JSON
                    String responseData = response.body().string();
                    System.out.println("服务器响应: " + responseData);
                    future.complete(responseData);

                    callback.onSuccess(responseData);

                } else {
                    future.complete(Constant.SERVER_ERROR);
                }
            }
        });
        return future.get();
    }
    // DetectEyes.java
    public  Bitmap cropEye(Bitmap source, String result) {

        String[] data = result.split(",");

        int x = Integer.parseInt(data[2]);
        int y = Integer.parseInt(data[3]);
        int w = Integer.parseInt(data[4]);
        int h = Integer.parseInt(data[5]);
        System.out.println(result);
        return Bitmap.createBitmap(source, x, y, w, h);
    }
}
