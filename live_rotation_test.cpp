#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <omp.h>

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// XYZ-eular rotation 
Mat eular2rot(Vec3d theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );
     
    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );
     
    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
     
    // Combined rotation matrix
    Mat R = R_x * R_y * R_z;
     
    return R;
}

// Rotation matrix to rotation vector in XYZ-eular order
Vec3d rot2eular(Mat R)
{
    double sy = sqrt(R.at<double>(2,2) * R.at<double>(2,2) +  R.at<double>(1,2) * R.at<double>(1,2) );
 
    bool singular = sy < 1e-6; // If
 
    double x, y, z;
    if (!singular)
    {
        x = atan2(-R.at<double>(1,2) , R.at<double>(2,2));
        y = atan2(R.at<double>(0,2), sy);
        z = atan2(-R.at<double>(0,1), R.at<double>(0,0));
    }
    else
    {
        x = 0;
        y = atan2(R.at<double>(0,2), sy);
        z = atan2(-R.at<double>(0,1), R.at<double>(0,0));
    }
    return Vec3d(x, y, z);
}

vector<Vec6d> pixel2cart(Mat& img)
{
    int width = img.cols;
    int height = img.rows;

    Vec3b* img_data = (Vec3b*)img.data;
    vector<Vec6d> vec_cartesian(width*height);

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            Vec2d vec_rad = Vec2d(M_PI*i/height, 2*M_PI*j/width);
            vec_cartesian[i*width + j][0] = sin(vec_rad[0])*cos(vec_rad[1]);
            vec_cartesian[i*width + j][1] = sin(vec_rad[0])*sin(vec_rad[1]);
            vec_cartesian[i*width + j][2] = cos(vec_rad[0]);
            vec_cartesian[i*width + j][3] = img_data[i*width + j][0];
            vec_cartesian[i*width + j][4] = img_data[i*width + j][1];
            vec_cartesian[i*width + j][5] = img_data[i*width + j][2];
        }
    }

    return vec_cartesian;
}

Mat rotate_cart(vector<Vec6d>& vec_cartesian, Mat& rot_mat, int width, int height)
{
    Mat rotated_img(height, width, CV_8UC3);

    double* rot_mat_data = (double*)rot_mat.data;
    Vec3b* rotated_img_data = (Vec3b*)rotated_img.data;

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            Vec2d vec_rad = Vec2d(M_PI*i/height, 2*M_PI*j/width);
            Vec3d vec_cartesian_rot;
            vec_cartesian_rot[0] = rot_mat_data[0]*vec_cartesian[i*width + j][0] + rot_mat_data[1]*vec_cartesian[i*width + j][1] + rot_mat_data[2]*vec_cartesian[i*width + j][2];
            vec_cartesian_rot[1] = rot_mat_data[3]*vec_cartesian[i*width + j][0] + rot_mat_data[4]*vec_cartesian[i*width + j][1] + rot_mat_data[5]*vec_cartesian[i*width + j][2];
            vec_cartesian_rot[2] = rot_mat_data[6]*vec_cartesian[i*width + j][0] + rot_mat_data[7]*vec_cartesian[i*width + j][1] + rot_mat_data[8]*vec_cartesian[i*width + j][2];

            Vec2d vec_rot;
            vec_rot[0] = acos(vec_cartesian_rot[2]);
            vec_rot[1] = atan2(vec_cartesian_rot[1], vec_cartesian_rot[0]);
            if(vec_rot[1] < 0)
                vec_rot[1] += M_PI*2;

            Vec2i vec_pixel;
            vec_pixel[0] = height*vec_rot[0]/M_PI;
            vec_pixel[1] = width*vec_rot[1]/(2*M_PI);

            rotated_img_data[vec_pixel[0]*width + vec_pixel[1]] = Vec3b(vec_cartesian[i*width + j][3], vec_cartesian[i*width + j][4], vec_cartesian[i*width + j][5]);
        }
    }

    return rotated_img;
}

static void on_trackbar( int, void* ){
}

int main(int argc, char** argv)
{
    Mat im = imread(argv[1]);
    Mat im_run;
    resize(im, im_run, Size(), 0.2, 0.2);

    vector<Vec6d> vec_cartesian = pixel2cart(im_run);

    namedWindow("test");

    int x, y, z;
	createTrackbar("x", "test", &x, 360, on_trackbar);
	createTrackbar("y", "test", &y, 360, on_trackbar);
	createTrackbar("z", "test", &z, 360, on_trackbar);

	setTrackbarPos("x", "test", 180);
	setTrackbarPos("y", "test", 180);
	setTrackbarPos("z", "test", 180);

    while(1)
    {
        Mat rot_mat = eular2rot(Vec3f(RAD(x-180), RAD(y-180), RAD(z-180)));
        Vec3d rot_vec = rot2eular(rot_mat);
        Mat im_rotate = rotate_cart(vec_cartesian, rot_mat, im_run.cols, im_run.rows);

        Mat show;
        resize(im_rotate, show, Size(1280, 640));
        string angle_vec_txt = to_string(DEGREE(rot_vec[0])) + ',' + to_string(DEGREE(rot_vec[1])) + ',' + to_string(DEGREE(rot_vec[2]));
        putText(show, angle_vec_txt, Point(0, show.rows/2), 2, 2, Scalar::all(255));
        imshow("test", show);
        waitKey(20);
    }
    
    return 0;
}