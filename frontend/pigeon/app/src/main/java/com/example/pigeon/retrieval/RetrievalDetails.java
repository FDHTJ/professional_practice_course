package com.example.pigeon.retrieval;

import android.app.Activity;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.bumptech.glide.Glide;
import com.bumptech.glide.load.DataSource;
import com.bumptech.glide.load.engine.GlideException;
import com.bumptech.glide.request.RequestListener;
import com.bumptech.glide.request.target.Target;
import com.example.pigeon.R;
import com.example.pigeon.utills.Constant;
import com.example.pigeon.utills.Pigeon;
import com.example.pigeon.utills.ViewImage;
import com.google.android.material.bottomsheet.BottomSheetDialog;
import com.google.android.material.floatingactionbutton.FloatingActionButton;

import java.util.Arrays;
import java.util.List;

import ai.z.openapi.ZhipuAiClient;
import ai.z.openapi.service.model.ChatCompletionCreateParams;
import ai.z.openapi.service.model.ChatCompletionResponse;
import ai.z.openapi.service.model.ChatMessage;
import ai.z.openapi.service.model.ChatMessageRole;
import ai.z.openapi.service.model.ImageUrl;
import ai.z.openapi.service.model.MessageContent;

public class RetrievalDetails extends AppCompatActivity {
    Boolean loadImage=false;
    String helperResult=null;
    public  String getHelperResult() {
        String apiKey = Constant.MODEL_API_KEY; // 请填写您自己的APIKey
        ZhipuAiClient client = ZhipuAiClient.builder()
                .apiKey(apiKey)
                .build();
        String prompt="你是一个资深的信鸽专家，你能够通过所给的信鸽的相关信息，精确的总结出信鸽的信息。\n" +
                "指令：给定以下信鸽的信息，生成关于该信鸽的总结描述，请尽可能的保持专业和精简，使用户能够在最短的时间内了解该信鸽。如果有附加的图片，请将图片中的信息一并总结（包括但不限于羽毛、瞳孔、性别等）。\n" +
                "信鸽信息：\n" +
                "姓名："+p.name+"\n"+
                "ID："+p.pId+"\n"+
                "省份："+p.province+"\n"+
                "城市："+p.city+"\n"+
                "详细介绍："+p.details+"\n";
        List<MessageContent> messageContentList;
        if (loadImage){
            messageContentList = Arrays.asList(
                    MessageContent.builder()
                            .type("text")
                            .text(prompt)
                            .build(),
                    MessageContent.builder()
                            .type("image_url")
                            .imageUrl(ImageUrl.builder()
                                    .url(p.image)
                                    .build())
                            .build());
        }else{
            messageContentList = Arrays.asList(
                    MessageContent.builder()
                            .type("text")
                            .text(prompt)
                            .build());
        }
        ChatCompletionCreateParams request = ChatCompletionCreateParams.builder()
                .model("glm-4.6v-flash")
                .messages(Arrays.asList(
                        ChatMessage.builder()
                                .role(ChatMessageRole.USER.value())
                                .content(messageContentList)
                                .build()))
                .build();

        ChatCompletionResponse response = client.chat().createChatCompletion(request);

        if (response.isSuccess()) {
            ChatMessage reply = response.getData().getChoices().get(0).getMessage();
            System.out.println(reply);

            return reply.getContent().toString();
        } else {
            System.err.println("错误: " + response.getMsg());
        }
        return "";
    }

    private void showBottomSheetDialog() {
        // 1. 创建 BottomSheetDialog
        BottomSheetDialog bottomSheetDialog = new BottomSheetDialog(this);

        // 2. 加载布局
        View view = getLayoutInflater().inflate(R.layout.helper_results, null);
        bottomSheetDialog.setContentView(view);

        // 3. 获取布局里的控件
        View loadingView = view.findViewById(R.id.helper_loading);
        View contentView = view.findViewById(R.id.helper_content);
        TextView detailText = view.findViewById(R.id.helper_text);

        // 4. 显示弹窗（自带底部滑出动画）
        bottomSheetDialog.show();

        // 为了让圆角背景生效，需要把默认的背景设为透明
        // 否则圆角外面会有白色的直角底色
        if (bottomSheetDialog.getWindow() != null) {
            bottomSheetDialog.getWindow().findViewById(com.google.android.material.R.id.design_bottom_sheet)
                    .setBackgroundResource(android.R.color.transparent);
        }


        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // 【子线程】执行耗时操作
                    String result;
                    if (helperResult==null) {
                        result = getHelperResult();
                        helperResult=result;
                    }
                    else{
                        result=helperResult;
                    }
                    // 网络请求结束，切换回【主线程】更新UI
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // 检查弹窗是否还显示着（防止用户提前关闭导致空指针或崩溃）
                            if (bottomSheetDialog.isShowing()) {
                                // 隐藏加载条，显示内容
                                loadingView.setVisibility(View.GONE);
                                contentView.setVisibility(View.VISIBLE);

                                // 处理返回结果为空的情况
                                if (result != null && !result.isEmpty()) {
                                    detailText.setText(result);
                                } else {
                                    detailText.setText("未能获取到信鸽信息，请重试。");
                                }
                            }
                        }
                    });

                } catch (Exception e) {
                    e.printStackTrace();
                    // 出错时也要切回主线程提示用户
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (bottomSheetDialog.isShowing()) {
                                loadingView.setVisibility(View.GONE);
                                contentView.setVisibility(View.VISIBLE);
                                detailText.setText("发生错误：" );
                            }
                        }
                    });
                }
            }
        }).start();
//        try {
//            String result=getHelperResult();
//
//            // 检查弹窗是否还显示着（防止用户提前关闭了）
//            if (bottomSheetDialog.isShowing()) {
//                // 隐藏加载条，显示内容
//                loadingView.setVisibility(View.GONE);
//                contentView.setVisibility(View.VISIBLE);
//                // 这里可以设置动态获取到的文字
//                detailText.setText(result);
//            }
//
//        }catch (Exception e){
//            e.printStackTrace();
//        }

    }
    Pigeon p;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.retrieval_details);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.retrieval_details), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        ImageView image = findViewById(R.id.detail_image);
        TextView name = findViewById(R.id.detail_name);
        TextView id = findViewById(R.id.detail_id);
        TextView province = findViewById(R.id.detail_province);
        TextView city = findViewById(R.id.detail_city);
        TextView intro = findViewById(R.id.detail_intro);
        p=(Pigeon) getIntent().getSerializableExtra("pigeon");
        FloatingActionButton helper=findViewById(R.id.detail_helper);
        helper.setOnTouchListener(new View.OnTouchListener() {
            float downX, downY;
            float viewX, viewY;
            boolean isDragging = false;
            final int TOUCH_SLOP = 10;

            @Override
            public boolean onTouch(View v, MotionEvent event) {

                View parent = (View) v.getParent();

                switch (event.getAction()) {

                    case MotionEvent.ACTION_DOWN:
                        downX = event.getRawX();
                        downY = event.getRawY();
                        viewX = v.getX();
                        viewY = v.getY();
                        isDragging = false;
                        return true;

                    case MotionEvent.ACTION_MOVE:
                        float dx = event.getRawX() - downX;
                        float dy = event.getRawY() - downY;

                        if (Math.abs(dx) > TOUCH_SLOP || Math.abs(dy) > TOUCH_SLOP) {
                            isDragging = true;
                        }

                        float targetX = viewX + dx;
                        float targetY = viewY + dy;

                        // 限制不出父布局
                        float maxX = parent.getWidth() - v.getWidth();
                        float maxY = parent.getHeight() - v.getHeight();

                        targetX = Math.max(0, Math.min(targetX, maxX));
                        targetY = Math.max(0, Math.min(targetY, maxY));

                        v.setX(targetX);
                        v.setY(targetY);
                        return true;

                    case MotionEvent.ACTION_UP:
                        if (!isDragging) {
                            v.performClick();
                            return true;
                        }

                        // ===== 自动吸边 =====
                        float centerX = v.getX() + v.getWidth() / 2f;
                        float parentCenterX = parent.getWidth() / 2f;

                        float finalX;
                        if (centerX < parentCenterX) {
                            finalX = 0; // 吸左
                        } else {
                            finalX = parent.getWidth() - v.getWidth(); // 吸右
                        }

                        v.animate()
                                .x(finalX)
                                .setDuration(200)
                                .start();

                        return true;
                }
                return false;
            }
        });
        helper.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showBottomSheetDialog();
            }
        });

        // 从 Intent 获取传递的数据
        String pigeonName = p.name;
        String pigeonId = p.pId;
        String pigeonProvince = p.province;
        String pigeonCity = p.city;
        String pigeonIntro = p.details;
        String imageUrl = p.image;

        // 设置文本
        name.setText("名字：" + pigeonName);
        id.setText("ID：" + pigeonId);
        province.setText("省份：" + pigeonProvince);
        city.setText("城市：" + pigeonCity);
        intro.setText(pigeonIntro);

        Glide.with(this).load(imageUrl).listener(new RequestListener<Drawable>() {
                    @Override
                    public boolean onLoadFailed(@Nullable GlideException e, @Nullable Object model, @NonNull Target<Drawable> target, boolean isFirstResource) {

                        return false;
                    }

                    @Override
                    public boolean onResourceReady(@NonNull Drawable resource, @NonNull Object model, Target<Drawable> target, @NonNull DataSource dataSource, boolean isFirstResource) {
                        loadImage=true;
                        return false;
                    }
                })
                .error(R.drawable.no_image)
                .fallback(R.drawable.no_image).into(image);
        image.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(!loadImage){
                    return;
                }
                Intent intent= new Intent(RetrievalDetails.this, ViewImage.class);
                intent.putExtra("image_url",imageUrl);
                startActivity(intent);
            }
        });

    }
}