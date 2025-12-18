package com.example.pigeon.comparison;
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
import android.widget.TextView;
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

import org.json.JSONObject;

import java.io.IOException;

public class Comparison extends AppCompatActivity {
    private String detectResult0 = Constant.FAIL_TO_DETECT_YOLO;
    private Bitmap bitmap0;
    private Bitmap bitmap1;
    private String detectResult1 = Constant.FAIL_TO_DETECT_YOLO;
    ImageView image0;
    ImageView image1;
    FrameLayout loadingOverlay;
    private Integer currentImageView = 0;
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
                            Toast.makeText(Comparison.this, Constant.FAIL_TO_DETECT_YOLO, Toast.LENGTH_SHORT).show();
                        } else {
                            if (currentImageView == 0) {
                                image0.setImageBitmap(bitmap);
                                detectResult0 = selectResult;
                                bitmap0 = mutableBitmap.copy(Bitmap.Config.ARGB_8888, true);
                            } else {
                                image1.setImageBitmap(bitmap);
                                detectResult1 = selectResult;
                                bitmap1 = mutableBitmap.copy(Bitmap.Config.ARGB_8888, true);
                            }
                        }

                    } catch (IOException e) {
                        e.printStackTrace();
                        Toast.makeText(Comparison.this, Constant.FAIL_TO_READ_IMAGE, Toast.LENGTH_SHORT).show();
                    }

                } else {
                    Toast.makeText(Comparison.this, Constant.FAIL_TO_READ_IMAGE, Toast.LENGTH_SHORT).show();

                }
            }
    );
    Button startComparison;
    TextView comparisonResult;
    ProgressBar comparisonProgressbar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.comparison);
        image0 = findViewById(R.id.comparison_upload1);
        image1 = findViewById(R.id.comparison_upload2);
        startComparison = findViewById(R.id.start_comparison);
        comparisonResult = findViewById(R.id.comparison_result);
        comparisonProgressbar = findViewById(R.id.comparison_progressbar);
        loadingOverlay=findViewById(R.id.comparison_loading_overlay);
        EdgeToEdge.enable(this);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.comparison_layout), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        image0.setOnClickListener(view -> {
            currentImageView = 0;
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

            // 启动相册！
            galleryLauncher.launch(intent);
        });

        image1.setOnClickListener(view -> {
            currentImageView = 1;
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

            // 启动相册！
            galleryLauncher.launch(intent);
        });
        startComparison.setOnClickListener(view -> {
            if (detectResult0.equals(Constant.FAIL_TO_DETECT_YOLO) || detectResult1.equals(Constant.FAIL_TO_DETECT_YOLO)) {
                Toast.makeText(Comparison.this, Constant.LACK_IMAGE, Toast.LENGTH_SHORT).show();

            }else{
                comparisonProgressbar.setVisibility(View.VISIBLE);
                loadingOverlay.setVisibility(View.VISIBLE);
                new Thread(()->{
                    Bitmap cropEye0 = DetectEyes.getInstance().cropEye(bitmap0, detectResult0);
                    Bitmap cropEye1 = DetectEyes.getInstance().cropEye(bitmap1, detectResult1);

                    try {
                        DetectEyes.getInstance().uploadImageToServerForComparison(cropEye0, cropEye1, new Callback() {
                                @Override
                                public void onSuccess(String result) {
                                    System.out.println(result);
                                    runOnUiThread(() -> {
                                        try{
                                        JSONObject jsonObject = new JSONObject(result);
                                        String state = jsonObject.getString("status");
                                        if (state.equals(Constant.SERVER_SUCCESS)) {
                                            System.out.println(state);
                                            comparisonResult.setText(new JSONObject(jsonObject.getString("data")).getString("result"));
                                            Toast.makeText(Comparison.this, Constant.SUCCESSFUL, Toast.LENGTH_SHORT).show();

                                        } else {
                                            Toast.makeText(Comparison.this, Constant.SERVER_ERROR, Toast.LENGTH_SHORT).show();
                                        }
                                        }
                                        catch (Exception e){
                                            Toast.makeText(Comparison.this, Constant.SERVER_ERROR, Toast.LENGTH_SHORT).show();
                                        }finally {
                                            comparisonProgressbar.setVisibility(View.INVISIBLE);
                                            loadingOverlay.setVisibility(View.INVISIBLE);
                                        }
                                    });

                                }

                                @Override
                                public void onError(Exception e) {
                                    e.printStackTrace();
                                    runOnUiThread(() -> {
                                        Toast.makeText(Comparison.this,
                                                Constant.SERVER_ERROR, Toast.LENGTH_SHORT).show();
                                        comparisonProgressbar.setVisibility(View.INVISIBLE);
                                        loadingOverlay.setVisibility(View.INVISIBLE);
                                    });
                                }
                            });
                    } catch (Exception e) {
                        e.printStackTrace();
                        runOnUiThread(() -> {
                            Toast.makeText(Comparison.this,
                                    Constant.SERVER_ERROR, Toast.LENGTH_SHORT).show();
                            comparisonProgressbar.setVisibility(View.INVISIBLE);
                            loadingOverlay.setVisibility(View.INVISIBLE);
                        });
                    }

                }).start();
            }
        });

    }


};
