package com.example.pigeon.retrieval;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.pigeon.R;
import com.example.pigeon.utills.Callback;
import com.example.pigeon.utills.Constant;
import com.example.pigeon.utills.DetectEyes;
import com.example.pigeon.utills.Pigeon;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;

public class Retrieval extends AppCompatActivity {
    ImageView upload;
    Button start;
    String detectResult= Constant.FAIL_TO_DETECT_YOLO;
    Bitmap bitmapResult;
    ProgressBar retrievalProgressBar;
    FrameLayout lodingOverlay;
    ArrayList<Pigeon> retrievedPigeons;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.retrieval);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.retrieval_layout), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        lodingOverlay=findViewById(R.id.retrieval_loading_overlay);
        upload=findViewById(R.id.retrieval_upload);
        retrievalProgressBar=findViewById(R.id.retrieval_progressbar);
        start=findViewById(R.id.start_retrieval);
        upload.setOnClickListener(view -> {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            galleryLauncher.launch(intent);
        });

        start.setOnClickListener(view -> {
            if (detectResult.equals(Constant.FAIL_TO_DETECT_YOLO)) {
                Toast.makeText(Retrieval.this, Constant.LACK_IMAGE, Toast.LENGTH_SHORT).show();

            }else{
                retrievalProgressBar.setVisibility(View.VISIBLE);
                lodingOverlay.setVisibility(View.VISIBLE);
                new Thread(()->{
                    Bitmap cropEye = DetectEyes.getInstance().cropEye(bitmapResult, detectResult);

                    try {
                        DetectEyes.getInstance().uploadImageToServerForRetrieval( cropEye, new Callback() {
                            @Override
                            public void onSuccess(String result) {
                                System.out.println(result);
                                runOnUiThread(() -> {
                                    try{
                                        JSONObject jsonObject = new JSONObject(result);
                                        String state = jsonObject.getString("status");
                                        if (state.equals(Constant.SERVER_SUCCESS)) {
                                            System.out.println(state);
                                            String jsonArr=new JSONObject(jsonObject.getString("data")).getString("result");
                                            JSONArray results=new JSONArray(jsonArr);
                                            retrievedPigeons=new ArrayList<Pigeon>();
                                            for(int i=0;i<results.length();i++){

                                                JSONObject j = results.getJSONObject(i);
                                                Pigeon p=new Pigeon();
                                                p.name=j.getString("name");
                                                p.details=j.getString("details");
                                                p.city=j.getString("city");
                                                p.province=j.getString("province");
                                                p.image=j.getString("image").trim();
                                                p.pId=j.getString("pg_id");
                                                retrievedPigeons.add(p);
                                            }
                                            Toast.makeText(Retrieval.this, Constant.SUCCESSFUL, Toast.LENGTH_SHORT).show();
                                            Intent intent=new Intent(Retrieval.this,RetrievalResult.class);
                                            intent.putExtra("pigeonList",retrievedPigeons);
                                            startActivity(intent);
                                        } else {
                                            Toast.makeText(Retrieval.this, Constant.SERVER_ERROR, Toast.LENGTH_SHORT).show();
                                        }
                                    }
                                    catch (Exception e){
                                        e.printStackTrace();
                                        Toast.makeText(Retrieval.this, Constant.SERVER_ERROR, Toast.LENGTH_SHORT).show();
                                    }finally {
                                        retrievalProgressBar.setVisibility(View.INVISIBLE);
                                        lodingOverlay.setVisibility(View.INVISIBLE);

                                    }
                                });

                            }

                            @Override
                            public void onError(Exception e) {
                                runOnUiThread(() -> {
                                    Toast.makeText(Retrieval.this,
                                            Constant.SERVER_ERROR, Toast.LENGTH_SHORT).show();
                                    retrievalProgressBar.setVisibility(View.INVISIBLE);
                                    lodingOverlay.setVisibility(View.INVISIBLE);
                                });
                            }
                        });
                    } catch (Exception e) {
                        e.printStackTrace();
                        runOnUiThread(() -> {
                            Toast.makeText(Retrieval.this,
                                    Constant.SERVER_ERROR, Toast.LENGTH_SHORT).show();
                            retrievalProgressBar.setVisibility(View.INVISIBLE);
                            lodingOverlay.setVisibility(View.INVISIBLE);
                        });
                    }

                }).start();


            }
        });

    }
    private final ActivityResultLauncher<Intent> galleryLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                // 判断用户是否真的选了图 (RESULT_OK)
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    // 获取图片的 Uri (类似于文件的路径)
                    Uri imageUri = result.getData().getData();
                    try {
                        Bitmap bitmap = DetectEyes.getInstance().getBitmapFromUri(imageUri, getContentResolver());
                        String yoloDetectResult = DetectEyes.getInstance().getYoloDetectResult(bitmap);
                        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                        String selectResult = DetectEyes.getInstance().drawRects(bitmap, yoloDetectResult);
                        if (selectResult.equals(Constant.FAIL_TO_DETECT_YOLO)) {
                            Toast.makeText(Retrieval.this, Constant.FAIL_TO_DETECT_YOLO, Toast.LENGTH_SHORT).show();
                        } else {
                                upload.setImageBitmap(bitmap);
                                detectResult = selectResult;
                                bitmapResult = mutableBitmap.copy(Bitmap.Config.ARGB_8888, true);
                        }

                    } catch (IOException e) {
                        e.printStackTrace();
                        Toast.makeText(Retrieval.this, Constant.FAIL_TO_READ_IMAGE, Toast.LENGTH_SHORT).show();
                    }

                } else {
                    Toast.makeText(Retrieval.this, Constant.FAIL_TO_READ_IMAGE, Toast.LENGTH_SHORT).show();
                }
            }
    );
}