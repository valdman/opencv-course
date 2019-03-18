#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using std::vector;
using std::cout;
using std::endl;
using std::string;

const int samples_count = 6;
const int slider_max = 100;

char windowName[50] = "Brightness Alteration";
char FirstTrackbarName[50] = "First parameter";
char SecondTrackbarName[50] = "Second parameter";
int a_slider = 1;
int b_slider = 1;

Mat grayscaleDefault;

static double transform(double x, double y, uchar z) {
  return a_slider * (sin(x / a_slider) + 1) + b_slider * (cos(y / b_slider) + 1) + z;
}

static Mat adjustAB(Mat& image) {
  int rows = image.rows;
  int cols = image.cols;
  Mat new_image = Mat::zeros(rows, cols, CV_8UC1);
  for( int y = 0; y < rows; y++ ) {
        for( int x = 0; x < cols; x++ ) {
            for( int c = 0; c < image.channels(); c++ ) {
              int newVal = saturate_cast<uchar>( transform(x, y, image.at<Vec3b>(y,x)[c]) );
              uchar& targetPixel = new_image.at<Vec3b>(y,x)[c];
              targetPixel = newVal;
            }
        }
    }

  return new_image;
}

static void onTrackbar(int, void *) {
  Mat newImage = adjustAB(grayscaleDefault);
  Mat out;
  hconcat(grayscaleDefault, newImage, out);
  imshow(windowName, out);
  waitKey();
}

static void configureGUI() {
  namedWindow(windowName, WINDOW_AUTOSIZE); // Create Window
  createTrackbar(FirstTrackbarName, windowName, &a_slider, slider_max, onTrackbar);
  createTrackbar(SecondTrackbarName, windowName, &b_slider, slider_max, onTrackbar);
  onTrackbar(0, 0);
}

int main( int argc, char** argv ) {
  CommandLineParser parser( argc, argv, "{@input | ./assets/chrome_logo.png | input image}" );
  Mat colored = imread(parser.get<string>("@input"));
  if (colored.empty()) {
    cout << "Error loading image" << endl;
    return -1;
  }

  cvtColor(colored, grayscaleDefault, COLOR_BGR2GRAY);
  configureGUI();
  
  waitKey();
  return 0;
}