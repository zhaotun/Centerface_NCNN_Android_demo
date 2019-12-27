#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>

// ncnn
#include "net.h"

#include "centerface.h"

using namespace std;
#define TAG "RetinafaceSo"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)

//定义一个全局的 Centerface 类，后续所有的操作都是针对这个类进行
static Centerface *centerface;

//sdk是否初始化成功
bool detection_sdk_init_ok = false;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_poya_centerface_Centerface_FaceDetectionModelInit(JNIEnv *env, jobject instance,
                                                jstring faceDetectionModelPath_) {
     LOGD("JNI开始人脸检测模型初始化，啦啦啦-----");
    //如果已初始化则直接返回
    if (detection_sdk_init_ok) {
        //  LOGD("人脸检测模型已经导入");
        return true;
    }
    
	jboolean tRet = false;
    if (NULL == faceDetectionModelPath_) {
        //   LOGD("导入的人脸检测的目录为空");
        return tRet;
    }

    //获取MTCNN模型的绝对路径的目录（不是/aaa/bbb.bin这样的路径，是/aaa/)
    const char *faceDetectionModelPath = env->GetStringUTFChars(faceDetectionModelPath_, 0);
    if (NULL == faceDetectionModelPath) {
        return tRet;
    }

    string tFaceModelDir = faceDetectionModelPath;
    string tLastChar = tFaceModelDir.substr(tFaceModelDir.length() - 1, 1);
    //LOGD("init, tFaceModelDir last =%s", tLastChar.c_str());
    //目录补齐/
    if ("\\" == tLastChar) {
        tFaceModelDir = tFaceModelDir.substr(0, tFaceModelDir.length() - 1) + "/";
    } else if (tLastChar != "/") {
        tFaceModelDir += "/";
    }
    LOGD("init, tFaceModelDir=%s", tFaceModelDir.c_str());

    //没判断是否正确导入，懒得改了
//    retinafacenet = new RetinaFaceNet(tFaceModelDir);
    centerface = new Centerface(tFaceModelDir);
    //centerface->init(tFaceModelDir);
    //mtcnn->SetMinFace(40);

    env->ReleaseStringUTFChars(faceDetectionModelPath_, faceDetectionModelPath);
    detection_sdk_init_ok = true;
    tRet = true;
    return tRet;
}

JNIEXPORT jboolean JNICALL
Java_com_poya_centerface_Centerface_SetThreadsNumber(JNIEnv *env, jobject instance, jint threadsNumber) {
    if(!detection_sdk_init_ok){
        LOGD("人脸检测MTCNN模型SDK未初始化，直接返回");
        return false;
    }

    if(threadsNumber!=1&&threadsNumber!=2&&threadsNumber!=4&&threadsNumber!=8){
        LOGD("线程只能设置1，2，4，8");
        return false;
    }

    centerface->SetNumThreads(threadsNumber);
    return  true;
}

JNIEXPORT jintArray JNICALL
Java_com_poya_centerface_Centerface_FaceDetect(JNIEnv *env, jobject instance, jbyteArray imageDate_,
                                    jint imageWidth, jint imageHeight, jint imageChannel) {
    //  LOGD("JNI开始检测人脸");
    if(!detection_sdk_init_ok){
        LOGD("人脸检测MTCNN模型SDK未初始化，直接返回空");
        return NULL;
    }

    int tImageDateLen = env->GetArrayLength(imageDate_);
    if(imageChannel == tImageDateLen / imageWidth / imageHeight){
        LOGD("数据宽=%d,高=%d,通道=%d",imageWidth,imageHeight,imageChannel);
    }
    else{
        LOGD("数据长宽高通道不匹配，直接返回空");
        return NULL;
    }

    jbyte *imageDate = env->GetByteArrayElements(imageDate_, NULL);
    if (NULL == imageDate){
        LOGD("导入数据为空，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    if(imageWidth<20||imageHeight<20){
        LOGD("导入数据的宽和高小于20，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    //TODO 通道需测试
    if(3 == imageChannel || 4 == imageChannel){
        //图像通道数只能是3或4；
    }else{
        LOGD("图像通道数只能是3或4，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    //int32_t minFaceSize=40;
    //mtcnn->SetMinFace(minFaceSize);

    unsigned char *faceImageCharDate = (unsigned char*)imageDate;
    ncnn::Mat ncnn_img;
    if(imageChannel==3) {
       ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_BGR2RGB,
                                                    imageWidth, imageHeight);
    }else{
        //ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_RGBA2RGB, imageWidth, imageHeight);
        ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_RGBA2RGB, imageWidth, imageHeight);
    }

    std::vector<FaceInfo> faces ;
    centerface->detect(ncnn_img, faces, imageWidth,imageHeight,0.5, 0.3);

    //int32_t num_face = static_cast<int32_t>(finalBbox.size());
    int32_t num_face = static_cast<int32_t>(faces.size());
    LOGD("检测到的人脸数目----：%d\n", num_face);

    int out_size = 1+num_face*14;
	
    //  LOGD("内部人脸检测完成,开始导出数据");
    int *faceInfo = new int[out_size];

    faceInfo[0] = num_face;

    for(int i=0;i<num_face;i++){
        faceInfo[14*i+1] = faces[i].x1;//left
        faceInfo[14*i+2] = faces[i].y1;//top
        faceInfo[14*i+3] = faces[i].x2;//right
        faceInfo[14*i+4] = faces[i].y2;//bottom
        for (int j =0;j<10;j++){
            faceInfo[14*i+5+j]=static_cast<int>(faces[i].landmarks[j]);
        }
    }

    jintArray tFaceInfo = env->NewIntArray(out_size);
    env->SetIntArrayRegion(tFaceInfo,0,out_size,faceInfo);
    LOGD("内部人脸检测完成,导出数据成功");
    delete[] faceInfo;
    env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
    return tFaceInfo;
}

}