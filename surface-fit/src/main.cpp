#include <iostream>
#include<sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

double a, b;
Mat imageGrey;

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

double modeSwitcher(std::vector<double> v, int mode) {
    auto x = v[0],
        y = v[1];
    switch (mode) {
    case 1:
        return a * rastrigin(v) + b;
    case 2:
        return a * 300000 * sin((x)/350.0) * cos((y)/350.0) + b;
    case 3:
        return 2555555 * (x*x/(a*a)+y*y/(b*b));
    case 4: 
        return a * imageGrey.at<uchar>(x, y) + b;
    default:
        throw new invalid_argument("ivalidd mode argument value");
    };
}

int main( int argc, char** argv ) {
    auto params =   "{image           | ./input.jpeg  | input image}"
                    "{mode            |1              | surface type (1-4)}"
                    "{a               |1              | first paramater of run}"
                    "{b               |1              | second paramater of run}"
                    "{write           |False          | write results to files}"
                    "{help h usage ?  |               | print this message   }";
    CommandLineParser parser( argc, argv, params);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }
    Mat colored = imread(parser.get<string>("image"));
    auto mode = parser.get<int>("mode");
    auto write = parser.get<bool>("write");
    a = parser.get<double>("a");
    b = parser.get<double>("b");

    if (colored.empty()) {
        cout << "Error loading image" << endl;
        return -1;
    }
    if(mode > 4) {
        cout << "Ivallid mode argument vaue (mode can be 1-4)" << endl;
        return -1;
    }

    Mat img {colored.rows, colored.cols, CV_8UC1, cv::Scalar(0)};
    imageGrey = img;
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
            auto val = modeSwitcher(v, mode); // here goes func we need to interpolate
            Z.at<double>(i*cols+j, 0) = val;
        }
    SVD x(X);
    Mat A;
    x.backSubst(Z,A);
    cout<<A;
    auto winSize = cv::Size(400, 300);
    const string original = "Orignal";
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
    moveWindow(simulated, winSize.width * 2, 0);
    resizeWindow(simulated, winSize);

    const string difference = "Difference";
    moveWindow(difference, 0, winSize.height * 2);
    resizeWindow(difference, winSize);

    if(!write) {
        imshow(original, img);
        imshow(simulated,background);
        imshow(difference, img-background);
        waitKey();
        return 0;
    }

    auto ext = ".jpg";
    imwrite(original + ext, img);
    imwrite(simulated + ext, background);
    imwrite(difference + ext, img-background);
    return 0;
}