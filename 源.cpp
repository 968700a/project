#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <random>

int main()
{
    // 打开摄像头
    cv::VideoCapture cap;
    cap.open(0);

    if (!cap.isOpened()) {
        std::cerr << "Couldn't open camera" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cap >> frame; // 读取一帧图像

    int width = frame.size().width;
    int height = frame.size().height;
    int yy = 21; // 初始Y坐标
    int d = 7;   // 圆的移动步长
    const double PI = std::atan(1.0) * 4;
    int new_xx;
    int new_yy;
    int new_angle;
    int framesWait = 10; // 碰撞检测间隔帧数
    double result = -1;

    // 随机生成初始X坐标和角度
    std::random_device seed;
    std::mt19937 gen{ seed() };
    std::uniform_int_distribution<int> dist{ 20, (width - 20) };
    int xx = dist(gen);
    std::uniform_int_distribution<int> dist2{ 35, 145 };
    int angle = dist2(gen);
    cv::RNG rng(12345);

    // 随机生成初始圆颜色
    std::random_device colorSeed;
    std::mt19937 colorGen{ colorSeed() };
    std::uniform_int_distribution<int> colorDist{ 0, 255 };
    cv::Scalar circleColor(colorDist(colorGen), colorDist(colorGen), colorDist(colorGen));

    // 创建背景减法器
    cv::Ptr<cv::BackgroundSubtractor> pBackSub;
    pBackSub = cv::createBackgroundSubtractorMOG2(500, 256, false);

    // 记录圆的轨迹
    std::vector<cv::Point> trajectory;

    for (;;) {
        cap >> frame;
        if (frame.empty()) break;

        // 图像镜像处理
        cv::Mat frame_flipped;
        cv::flip(frame, frame_flipped, 1);

        cv::Mat frame_flipped2;
        cv::flip(frame, frame_flipped2, 1);

        // 转为灰度图
        cv::Mat greyMat;
        cv::cvtColor(frame_flipped, greyMat, cv::COLOR_BGR2GRAY);
        // 模糊处理
        cv::blur(greyMat, greyMat, cv::Size(3, 3));

        // 背景分离
        cv::Mat fgMask;
        pBackSub->apply(greyMat, fgMask, 0.75);

        imshow("Background Removal", fgMask);

        // 形态学操作
        int morph_size = 2;
        int elType = 0;
        cv::Mat element = cv::getStructuringElement(elType, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, element);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, element);

        // 轮廓检测与凸包计算
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(fgMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<cv::Point>> hull(contours.size());
        for (size_t i = 0; i < contours.size(); i++) {
            convexHull(cv::Mat(contours[i]), hull[i], true);
        }

        std::vector<std::vector<cv::Vec4i>> defects(contours.size());
        for (size_t i = 0; i < contours.size(); i++) {
            cv::Scalar color = cv::Scalar(128, 255, 8);
            drawContours(frame_flipped2, hull, (int)i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
        }

        // 碰撞检测
        if (framesWait <= 0) {
            for (int i = 0; i < hull.size(); ++i) {
                for (int j = 0; j <= 360; j = j + 5) {
                    new_angle = j;
                    new_xx = xx + (d * cos(new_angle * PI / 180));
                    new_yy = yy + (d * sin(new_angle * PI / 180));
                    result = cv::pointPolygonTest(hull[i], cv::Point2f((float)new_xx, (float)new_yy), false);
                    if (result >= 0) {
                        angle = (180 + new_angle);
                        framesWait = 10;
                        // 碰撞后随机改变颜色
                        circleColor = cv::Scalar(colorDist(colorGen), colorDist(colorGen), colorDist(colorGen));
                        result = -1;
                        goto contourBreak;
                    }
                }
            }
        }

    contourBreak:

        // 画圆
        cv::circle(frame_flipped, cv::Point(xx, yy), 20, circleColor, 2);
        trajectory.push_back(cv::Point(xx, yy));

        // 绘制轨迹
        for (size_t i = 1; i < trajectory.size(); ++i) {
            cv::line(frame_flipped, trajectory[i - 1], trajectory[i], cv::Scalar(0, 255, 0), 2);
        }

        imshow("After Morphology", fgMask);

        // 计算圆的下一个位置
        xx = xx + (d * cos(angle * PI / 180));
        yy = yy + (d * sin(angle * PI / 180));

        // 碰到边界时反弹
        if (xx <= 20) angle = (180 - angle), xx = xx + d;
        if (xx >= width - 20) angle = (180 - angle), xx = xx - d;
        if (yy >= height - 20) angle = (360 - angle), yy = yy - d;
        if (yy <= 20) angle = (360 - angle), yy = yy + d;

        // 如果发生碰撞，改变方向
        if (result >= 0) angle = (180 - angle);

        // 显示结果
        cv::imshow("Contours", frame_flipped2);
        cv::imshow("Final Output", frame_flipped);

        // 每33毫秒显示一帧（30帧/秒）
        if ((char)cv::waitKey(33) >= 0) break;

        // 碰撞检测间隔
        framesWait = framesWait - 1;
    }
    return 0;
}

