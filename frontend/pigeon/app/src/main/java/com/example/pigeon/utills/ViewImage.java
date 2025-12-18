package com.example.pigeon.utills;

import android.content.ContentValues;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.bumptech.glide.Glide;
import com.example.pigeon.R;
import com.github.chrisbanes.photoview.PhotoView;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;

public class ViewImage extends AppCompatActivity {
    private String imageUrl;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_view_image);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.view_image), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        // 获取传递过来的图片地址
        imageUrl = getIntent().getStringExtra("image_url");

        PhotoView photoView = findViewById(R.id.photo_view);

        // 1. 使用 Glide 加载图片
        Glide.with(this)
                .load(imageUrl)
                .into(photoView);

        // 2. 设置长按监听
        photoView.setOnLongClickListener(v -> {
            showDownloadDialog();
            return true;
        });
    }
    // 显示确认下载的弹窗
    private void showDownloadDialog() {
        new AlertDialog.Builder(this)
                .setTitle("提示")
                .setMessage("保存图片到相册吗？")
                .setPositiveButton("保存", (dialog, which) -> downloadImage())
                .setNegativeButton("取消", null)
                .show();
    }

    // 核心：下载图片逻辑
    private void downloadImage() {
        new Thread(() -> {
            try {
                // 利用 Glide 获取 Bitmap (必须在子线程执行)
                Bitmap bitmap = Glide.with(ViewImage.this)
                        .asBitmap()
                        .load(imageUrl)
                        .submit()
                        .get();

                // 保存到相册
                saveBitmapToGallery(bitmap);

            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> Toast.makeText(ViewImage.this, "下载失败", Toast.LENGTH_SHORT).show());
            }
        }).start();
    }

    // 保存 Bitmap 到系统图库 (适配 Android 10+ Scoped Storage)
    private void saveBitmapToGallery(Bitmap bitmap) {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, "pigeon_" + System.currentTimeMillis() + ".jpg");
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/MyPigeonApp"); // 相册中的文件夹名

        Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        try {
            if (uri != null) {
                OutputStream outputStream = getContentResolver().openOutputStream(uri);
                if (outputStream != null) {
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
                    outputStream.close();
                    runOnUiThread(() -> Toast.makeText(ViewImage.this, "图片已保存到相册", Toast.LENGTH_SHORT).show());
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            runOnUiThread(() -> Toast.makeText(ViewImage.this, "保存出错", Toast.LENGTH_SHORT).show());
        }
    }
}