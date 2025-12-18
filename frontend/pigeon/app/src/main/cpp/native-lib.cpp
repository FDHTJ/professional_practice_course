//#include <jni.h>
//#include <string>
//
//extern "C" JNIEXPORT jstring JNICALL
//Java_com_example_pigeon_MainActivity_stringFromJNI(
//        JNIEnv* env,
//        jobject /* this */) {
//    std::string hello = "Hello from C++";
//    return env->NewStringUTF(hello.c_str());
//}

#include <jni.h>
#include <string>
#include <vector>
#include <android/bitmap.h>
#include <android/log.h>
#include <android/asset_manager_jni.h>

// NCNN 头文件
#include "ncnn/net.h"
#include "ncnn/benchmark.h"

#define LOG_TAG "PigeonYolo"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// 定义一个结构体用来存检测结果
struct Object {
    float x, y, w, h;
    int label;
    float prob;
};

// 全局的模型实例
static ncnn::Net yolov11;
// 你的模型里的类别标签（根据你训练的模型修改，这里默认写几个示例）
static const char* class_names[] = {"eye"};

// --- 辅助函数：快速排序用于 NMS ---
static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;
    while (i <= j) {
        while (faceobjects[i].prob > p) i++;
        while (faceobjects[j].prob < p) j--;
        if (i <= j) {
            std::swap(faceobjects[i], faceobjects[j]);
            i++; j--;
        }
    }
    if (left < j) qsort_descent_inplace(faceobjects, left, j);
    if (i < right) qsort_descent_inplace(faceobjects, i, right);
}

// --- 辅助函数：计算 IOU (交并比) ---
static inline float intersection_area(const Object& a, const Object& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    if (x1 >= x2 || y1 >= y2) return 0.f;
    return (x2 - x1) * (y2 - y1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    int n = faceobjects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = faceobjects[i].w * faceobjects[i].h;

    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold) keep = 0;
        }
        if (keep) picked.push_back(i);
    }
}

// ==========================================
// 1. 初始化模型
// ==========================================
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_pigeon_utills_DetectEyes_initYolo(JNIEnv* env, jobject, jobject assetManager) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    yolov11.opt.use_vulkan_compute = false; // 开启 GPU 加速
    yolov11.opt.num_threads = 4;           // 线程数

    // 加载我们在 assets 里的文件
    int ret = yolov11.load_param(mgr, "yolov11n.param");
    if (ret != 0) {
        LOGD("加载 param 失败");
        return false;
    }

    ret = yolov11.load_model(mgr, "yolov11n.bin");
    if (ret != 0) {
        LOGD("加载 bin 失败");
        return false;
    }

    LOGD("YOLOv11 初始化成功！");
    return true;
}

// ==========================================
// 2. 执行检测
// ==========================================
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_pigeon_utills_DetectEyes_detect(JNIEnv* env, jobject, jobject bitmap) {
    // 1. 获取 Bitmap 信息
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;

    // 2. 图片转换：Bitmap -> NCNN Mat
    // 假设输入是 RGB_565 或者 RGBA_8888，NCNN 会自动处理
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, 640, 640);

    // 3. 归一化 (YOLOv8/11 通常是 0-1)
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // 4. 推理
    ncnn::Extractor ex = yolov11.create_extractor();
    ex.input("in0", in); // 这里的名字必须和 param 文件里的 Input 名字一致

    ncnn::Mat out;
    ex.extract("out0", out); // 这里的名字必须和 param 文件里的 Output 名字一致

    // 5. 解析输出 [1, 84, 8400]
    // out.w = 8400 (anchors), out.h = 84 (4 coords + 80 classes)

    std::vector<Object> proposals;
    const int num_class = 1; // 假设是 COCO 80类，如果是你自己的模型，请修改这里！

    // 遍历所有 8400 个 anchor
    for (int i = 0; i < out.w; i++) {
        float max_score = 0.f;
        int max_label = -1;

        // 找概率最大的类别 (从第 4 行开始是类别概率)
        for (int c = 0; c < num_class; c++) {
            float score = out.row(c + 4)[i];
            if (score > max_score) {
                max_score = score;
                max_label = c;
            }
        }

        if (max_score > 0.439f) { // 阈值过滤
            Object obj;
            // 获取坐标 (cx, cy, w, h)
            float cx = out.row(0)[i];
            float cy = out.row(1)[i];
            float w = out.row(2)[i];
            float h = out.row(3)[i];

            // 还原到 640x640 下的左上角坐标
            float x = cx - w * 0.5f;
            float y = cy - h * 0.5f;

            // 还原到原图尺寸
            obj.x = x * ((float)width / 640.f);
            obj.y = y * ((float)height / 640.f);
            obj.w = w * ((float)width / 640.f);
            obj.h = h * ((float)height / 640.f);
            obj.label = max_label;
            obj.prob = max_score;
            proposals.push_back(obj);
        }
    }
    if (proposals.empty()){
        return env->NewStringUTF("");
    }
    // 6. NMS (非极大值抑制) - 去掉重叠的框
    qsort_descent_inplace(proposals, 0, proposals.size() - 1);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, 0.439f);

    // 7. 拼接结果字符串返回给 Java (简单起见，返回一个字符串："label:prob:x:y:w:h|...")
    std::string result_str = "";
    int count = 0;
    for (int i : picked) {
        const Object& obj = proposals[i];

        // 拼接字符串
        result_str += std::to_string(obj.label) + "," +
                      std::to_string(obj.prob) + "," +
                      std::to_string((int)obj.x) + "," +
                      std::to_string((int)obj.y) + "," +
                      std::to_string((int)obj.w) + "," +
                      std::to_string((int)obj.h) + "|";
        count++;
    }

    LOGD("检测到 %d 个物体", count);
    return env->NewStringUTF(result_str.c_str());
}

