#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <random>

int main()
{
    // ������ͷ
    cv::VideoCapture cap;
    cap.open(0);

    if (!cap.isOpened()) {
        std::cerr << "Couldn't open camera" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cap >> frame; // ��ȡһ֡ͼ��

    int width = frame.size().width;
    int height = frame.size().height;
    int yy = 21; // ��ʼY����
    int d = 7;   // Բ���ƶ�����
    const double PI = std::atan(1.0) * 4;
    int new_xx;
    int new_yy;
    int new_angle;
    int framesWait = 10; // ��ײ�����֡��
    double result = -1;

    // ������ɳ�ʼX����ͽǶ�
    std::random_device seed;
    std::mt19937 gen{ seed() };
    std::uniform_int_distribution<int> dist{ 20, (width - 20) };
    int xx = dist(gen);
    std::uniform_int_distribution<int> dist2{ 35, 145 };
    int angle = dist2(gen);
    cv::RNG rng(12345);

    // ������ɳ�ʼԲ��ɫ
    std::random_device colorSeed;
    std::mt19937 colorGen{ colorSeed() };
    std::uniform_int_distribution<int> colorDist{ 0, 255 };
    cv::Scalar circleColor(colorDist(colorGen), colorDist(colorGen), colorDist(colorGen));

    // ��������������
    cv::Ptr<cv::BackgroundSubtractor> pBackSub;
    pBackSub = cv::createBackgroundSubtractorMOG2(500, 256, false);

    // ��¼Բ�Ĺ켣
    std::vector<cv::Point> trajectory;

    for (;;) {
        cap >> frame;
        if (frame.empty()) break;

        // ͼ������
        cv::Mat frame_flipped;
        cv::flip(frame, frame_flipped, 1);

        cv::Mat frame_flipped2;
        cv::flip(frame, frame_flipped2, 1);

        // תΪ�Ҷ�ͼ
        cv::Mat greyMat;
        cv::cvtColor(frame_flipped, greyMat, cv::COLOR_BGR2GRAY);
        // ģ������
        cv::blur(greyMat, greyMat, cv::Size(3, 3));

        // ��������
        cv::Mat fgMask;
        pBackSub->apply(greyMat, fgMask, 0.75);

        imshow("Background Removal", fgMask);

        // ��̬ѧ����
        int morph_size = 2;
        int elType = 0;
        cv::Mat element = cv::getStructuringElement(elType, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, element);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, element);

        // ���������͹������
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

        // ��ײ���
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
                        // ��ײ������ı���ɫ
                        circleColor = cv::Scalar(colorDist(colorGen), colorDist(colorGen), colorDist(colorGen));
                        result = -1;
                        goto contourBreak;
                    }
                }
            }
        }

    contourBreak:

        // ��Բ
        cv::circle(frame_flipped, cv::Point(xx, yy), 20, circleColor, 2);
        trajectory.push_back(cv::Point(xx, yy));

        // ���ƹ켣
        for (size_t i = 1; i < trajectory.size(); ++i) {
            cv::line(frame_flipped, trajectory[i - 1], trajectory[i], cv::Scalar(0, 255, 0), 2);
        }

        imshow("After Morphology", fgMask);

        // ����Բ����һ��λ��
        xx = xx + (d * cos(angle * PI / 180));
        yy = yy + (d * sin(angle * PI / 180));

        // �����߽�ʱ����
        if (xx <= 20) angle = (180 - angle), xx = xx + d;
        if (xx >= width - 20) angle = (180 - angle), xx = xx - d;
        if (yy >= height - 20) angle = (360 - angle), yy = yy - d;
        if (yy <= 20) angle = (360 - angle), yy = yy + d;

        // ���������ײ���ı䷽��
        if (result >= 0) angle = (180 - angle);

        // ��ʾ���
        cv::imshow("Contours", frame_flipped2);
        cv::imshow("Final Output", frame_flipped);

        // ÿ33������ʾһ֡��30֡/�룩
        if ((char)cv::waitKey(33) >= 0) break;

        // ��ײ�����
        framesWait = framesWait - 1;
    }
    return 0;
}

