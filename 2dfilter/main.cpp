#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

const uint W = 100;
const uint H = 100;

void initImage(const cv::Mat img)
{
    cv::Mat rc {W, H, CV_8U, cv::Scalar(0)};
    cv::circle(rc, Point(rc.rows/2, rc.cols/2), rc.cols/2, cv::Scalar(255), -1);
    cv::Rect r (0, 0, rc.cols, rc.rows);
    copyTo(rc, img(r), cv::Mat());

    rc = Mat{W, H, CV_8U, cv::Scalar(127)};
    cv::circle(rc, cv::Point(rc.rows/2, rc.cols/2), rc.cols/2, cv::Scalar(0), -1);
    r.x += rc.cols;
    copyTo(rc, img(r), cv::Mat());

    rc = cv::Mat{W, H, CV_8U, cv::Scalar(255)};
    cv::circle(rc, cv::Point(rc.rows/2, rc.cols/2), rc.cols/2, cv::Scalar(127), -1);
    r.x += rc.cols;
    copyTo(rc, img(r), cv::Mat());

    rc = cv::Mat{W, H, CV_8U, Scalar(255)};
    cv::circle(rc, cv::Point(rc.rows/2, rc.cols/2), rc.cols/2, cv::Scalar(0), -1);
    r.y += rc.rows;
    r.x = 0;
    copyTo(rc, img(r), cv::Mat());

    rc = cv::Mat{W, H, CV_8U, cv::Scalar(0)};
    cv::circle(rc, cv::Point(rc.rows/2, rc.cols/2), rc.cols/2, cv::Scalar(127), -1);
    r.x += rc.cols;
    copyTo(rc, img(r), cv::Mat());

    rc = cv::Mat{W, H, CV_8U, cv::Scalar(127)};
    cv::circle(rc, cv::Point(rc.rows/2, rc.cols/2), rc.cols/2, cv::Scalar(255), -1);
    r.x += rc.cols;
    copyTo(rc, img(r), cv::Mat());
}

int main()
{
  cv::Mat img {W * 2, H * 3, CV_8U, cv::Scalar(0)};
  initImage(img);
  cv::imshow("orig", img);
  cv::moveWindow("orig", 75, 75);

  cv::Mat_<double> kernel_core(3,3);
  kernel_core << 1.4,2,4,6,1,4,3,2,1;

  cv::Mat dst;
  cv::filter2D(img, dst, CV_8U, kernel_core);
  cv::imshow("filtered", dst);
  cv::waitKey();
}