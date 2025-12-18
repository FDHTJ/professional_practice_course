package com.example.pigeon;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import com.example.pigeon.comparison.Comparison;
import com.example.pigeon.databinding.ActivityMainBinding;
import com.example.pigeon.retrieval.Retrieval;
import com.example.pigeon.utills.DetectEyes;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'pigeon' library on application startup.
    static {
        System.loadLibrary("pigeon");
    }
    protected Button comparison;
    protected Button retrieval;
    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        comparison=findViewById(R.id.comparison);
        retrieval=findViewById(R.id.retrieval);
        EdgeToEdge.enable(this);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main_layout), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        boolean initYolo = DetectEyes.getInstance().initYolo(getAssets());//初始化yolo
        comparison.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, Comparison.class);
                startActivity(intent);
            }
        });

        retrieval.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, Retrieval.class);
                startActivity(intent);
            }
        });


    }


}


//    DetectEyes detector = DetectEyes.getInstance();
//        InputStream is = null;
//        try {
//            is = getAssets().open("img.png");
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }
//        Bitmap bitmap = BitmapFactory.decodeStream(is);
//        // 现在的 bitmap 是不可修改的，我们需要复制一份可修改的配置来画框
//        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
//
//        // 3. 执行检测
//        String result = detector.getYoloDetectResult(mutableBitmap);
//        String drawResult = detector.drawRects(mutableBitmap, result);
//        if (drawResult.equals(Constant.FAIL_TO_DETECT_YOLO) ){
//            Toast.makeText(this, drawResult, Toast.LENGTH_SHORT).show();
//        }
//        System.out.println("检测结果: " + result);
//        // 5. 显示到界面上 (假设你的布局里有个 ImageView 叫 imageView)
//        ImageView iv = findViewById(R.id.show_image);
//        iv.setImageBitmap(mutableBitmap);