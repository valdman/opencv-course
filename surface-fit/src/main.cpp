#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

double rastrigin(std::vector<double> argument, double slow = 1) {
   double evaluation = 0.0;

   int numberOfVariables = argument.size();

   const double pi = 3.1415927;

   evaluation = 10.0*numberOfVariables;

   for(int i = 0; i < numberOfVariables; i++)
   {
      evaluation += pow(argument[i] / slow, 2) - 10.0*cos(2.0*pi*argument[i]/slow);
   }

   return(evaluation);
}

double win(std::vector<double> argument) {
    auto reduction = true;
    for(auto x : argument) {
        auto a = x < 600;
        auto b = x > 300;
        reduction = reduction && a && b;
    }
    return reduction ? 255 : 0;
}

int main( int argc, char** argv ) {
    CommandLineParser parser( argc, argv, "{@input | ./tram.jpeg | input image}" );
    Mat colored = imread(parser.get<string>("@input"));
    if (colored.empty()) {
        cout << "Error loading image" << endl;
        return -1;
    }

    Mat img {colored.rows, colored.cols, CV_8UC1, cv::Scalar(0)};
    int rows = img.rows, cols = img.cols;
    cvtColor(colored, img, COLOR_BGR2GRAY);

    Mat X = Mat_<double>(rows*cols,6);
    Mat Z=Mat_<double>(rows*cols,1);
    for (int i=0;i<rows;i++)
        for (int j = 0; j < cols; j++)
        {
            double x =(j - cols/2)/double(cols),y = (i - rows/2)/double(rows);
            X.at<double>(i*cols+j, 0) = x*x;
            X.at<double>(i*cols+j, 1) = y*y;
            X.at<double>(i*cols+j, 2) = x*y;
            X.at<double>(i*cols+j, 3) = x;
            X.at<double>(i*cols+j, 4) = y;
            X.at<double>(i*cols+j, 5) = 1;
            std::vector v{x, y};
            auto val = 25 * rastrigin(v); // here goes func we need to interpolate
            Z.at<double>(i*cols+j, 0) = val;
        }
    SVD x(X);
    Mat A;
    x.backSubst(Z,A);
    cout<<A;
    auto winSize = cv::Size(400, 300);
    const string original = "Orignal";
    imshow(original, img);
    resizeWindow(original, winSize);
    cout<<A.at<double>(2,0);
    Mat background(rows,cols,CV_8UC1);
    for (int i=0;i<rows;i++)
        for (int j = 0; j < cols; j++)
        {
            double x=(j - cols / 2) / double(cols),y= (i - rows / 2) / double(rows);
            double quad=A.at<double>(0,0)*x*x+A.at<double>(1,0)*y*y+A.at<double>(2,0)*x*y;
            quad+=A.at<double>(3,0)*x+A.at<double>(4,0)*y+A.at<double>(5,0);
            background.at<uchar>(i,j) = saturate_cast<uchar>(quad);
        }
    const string simulated = "Simulated background";
    imshow(simulated,background);
    moveWindow(simulated, winSize.width * 2, 0);
    resizeWindow(simulated, winSize);

    const string difference = "Difference";
    imshow(difference, img-background);
    moveWindow(difference, 0, winSize.height * 2);
    resizeWindow(difference, winSize);

    waitKey();
    return 0;
}