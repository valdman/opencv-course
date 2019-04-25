#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

const int mattype = CV_8UC1;
const int PolyOrderX = 3;
const int PolyOrderY = 3;

char windowName[50] = "Polynomial surface fit";

cv::Mat SurfaceFit( cv::Mat& TheImage ) {
    int Nrows = TheImage.rows;
    int Ncols = TheImage.cols;
    int ImageType = TheImage.type();
    int Npixels = Nrows*Ncols;
    int n=0;
    int r, c, cnt, i, j, k, MatVal, nCol;
    int Dim1, Dim2;
    int PO_2xp1 = cv::max((2 * PolyOrderX + 1), (2 * PolyOrderY + 1));
    uint matSize = (PolyOrderX+1)*(PolyOrderY+1);

    // Create the x, y, and z arrays from which the image to be fitted
    double X[Npixels];
    double Y[Npixels];
    double Z[Npixels];
    cnt = 0;
    for(r=0; r<Nrows; r++) {
        for(c=0; c<Ncols; c++) {
            X[cnt] = c;
            Y[cnt] = r;
            Z[cnt] = TheImage.at<uchar>(r, c);
            cnt++;
        }
    }

	double XY_mat[matSize][matSize];
    int nnx[matSize][matSize];
    int nny[matSize][matSize];
    int aRow[matSize];

    // Create all the possible sums, Sxyz[][][]
    double Sxyz[PO_2xp1][PO_2xp1][2];
    double x, y, z;
    double powx, powy, powz;
    int nx, ny, nz;
    // Initialize all of the sums to zero
    for(nx=0; nx<PO_2xp1; nx++) {
        for(ny=0; ny<PO_2xp1; ny++) {
            for(nz=0; nz<2; nz++) {
                Sxyz[nx][ny][nz] = 0.0;
            }
        }
    }
    // Produce the sums
    for( i=0; i<Npixels; i++) {
        x = X[i]; y = Y[i]; z = Z[i];
        for(nx=0; nx<PO_2xp1; nx++) {
            powx = cv::pow(x,(double)nx);
            for(ny=0; ny<PO_2xp1; ny++) {
                powy = cv::pow(y,(double)ny);
                for(nz=0; nz<2; nz++) {
                    powz = cv::pow(z,(double)nz);
                    Sxyz[nx][ny][nz] += powx * powy * powz;
                }
            }
        }
    }

    // Create the patterns of "powers" for the X (horizontal) pixel indices
    int iStart = 2 * PolyOrderX;
    Dim1 = 0;
    while(Dim1<matSize) {
        for(i=0; i<(PolyOrderY+1); i++) {
            // A row of nnx[][] consists of an integer that starts with a value iStart and
            //  1) is repeated (PolyOrderX+1) times
            //  2) decremented by 1
            //  3) Repeat steps 1 and 2 for a total of (PolyOrderY+1) times
            nCol = 0;
            for(j=0; j<(PolyOrderX+1); j++ ) {
                for(k=0; k<(PolyOrderY+1); k++) {
                    aRow[nCol] = iStart - j;
                    nCol++;
                }
            }
            // Place this row into the nnx matrix
            for(Dim2=0; Dim2<matSize; Dim2++ ) {
                nnx[Dim1][Dim2] = aRow[Dim2];
            }
            Dim1++;
        }
        iStart--;
    }
    
    // Create the patterns of "powers" for the Y (vertical) pixel indices
    Dim1 = 0;
    while(Dim1<matSize) {
        iStart = 2 * PolyOrderY;
        for(i=0; i<(PolyOrderY+1); i++) {
            // A row of nny[][] consists of an integer that starts with a value iStart and
            //  1) place in matrix
            //  2) decremented by 1
            //  3) 1 thru 2 are repeated for a total of (PolyOrderX+1) times
            //  4) 1 thru 3 are repeat a total of (PolyOrderY+1) times
            nCol = 0;
            for(j=0; j<(PolyOrderX+1); j++ ) {
                for(k=0; k<(PolyOrderY+1); k++) {
                    aRow[nCol] = iStart - k;
                    nCol++;
                }
            }
            // Place this row into the nnx matrix
            for(Dim2=0; Dim2<matSize; Dim2++ ) {
                nny[Dim1][Dim2] = aRow[Dim2];
            }
            Dim1++;
            iStart--;
        }
    }

    // Put together the [XY] matrix
	for(r=0; r<matSize; r++) {
		for(c=0; c<matSize; c++) {
			nx = nnx[r][c];
			ny = nny[r][c];
			XY_mat[r][c] = Sxyz[nx][ny][0];
		}
	}

    // Put together the [Z] vector
	double Z_mat[matSize];
    c = 0;
    for(i=PolyOrderX; i>=0; i--) {
		for(j=PolyOrderY; j>=0; j--) {
            Z_mat[c] = Sxyz[i][j][1];
            c++;
        }
    }

    // Solve the linear system [XY] [P] = [Z] using the Jama.Matrix routines
	// 	[A_mat] [x_vec] = [b_vec]
	// (see example at   http://math.nist.gov/javanumerics/jama/doc/Jama/Matrix.html)
	cv::Mat A_mat {2, 5, mattype, XY_mat};
	cv::Mat b_vec {1, 10, mattype, Z_mat};
	cv::Mat x_vec;
  cv::solve(A_mat, b_vec, x_vec);

	// Place the Least Squares Fit results into the array Pfit
	double Pfit[matSize];
	for(i=0; i<matSize; i++) {
		Pfit[i] = x_vec.at<double>(i, 0);
	}

	// Reformat the results into a 2-D array where the array indices
    // specify the power of pixel indices.  For example,
    // z =    (G[2][3] y^2 + G[1][3] y^1 + G[0][3] y^0) x^3
    //      + (G[2][2] y^2 + G[1][2] y^1 + G[0][2] y^0) x^2
    //      + (G[2][1] y^2 + G[1][1] y^1 + G[0][1] y^0) x^1
    //      + (G[2][0] y^2 + G[1][0] y^1 + G[0][0] y^0) x^0
    cv:Mat Gfit{PolyOrderY + 1, PolyOrderX + 1, mattype, Scalar(0)};
    c = 0;
    for(i=PolyOrderX; i>=0; i--) {
		for(j=PolyOrderY; j>=0; j--) {
            Gfit.at<double>(j, i) = Pfit[c];
            c++;
        }
    }
    return Gfit ;
}


int main( int argc, char** argv ) {
  CommandLineParser parser( argc, argv, "{@input | ./tram.jpg | input image}" );
  Mat colored = imread(parser.get<string>("@input"));
  if (colored.empty()) {
    cout << "Error loading image" << endl;
    return -1;
  }

  Mat grayscaleDefault;
  cvtColor(colored, grayscaleDefault, COLOR_BGR2GRAY);
  imshow(windowName, SurfaceFit(grayscaleDefault));
  waitKey();
  return 0;
}