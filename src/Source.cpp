#include<iostream>
#include "opencv2\opencv.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#define AOIPointsNum 4
using namespace cv;
using namespace std;

// There should be a global image variable that could be used for displaying the final image after all the operations

Mat AOI_out;
vector<Point> AOI;
bool ready = false;

void callBackFunction(int event, int x, int y, int flags, void* userdata ) {
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		AOI.push_back(Point(x, y));
		if (AOI.size() > AOIPointsNum) {
			imwrite("C:\\Users\\faraz.bhatti\\Documents\\Visual Studio 2015\\Projects\\OpenCVIntro\\template.png", AOI_out);
			ready = true;
			AOI.clear();
		}
	}
	
}

void getTemplate(Mat image) { // Try and pass pointers
	Rect bounder(AOI[0], AOI[3]); // Fix these magic numbers
	AOI_out = image(bounder);
	imshow("Template", AOI_out);
}

void templateMatcher(Mat *image, Mat *temp) {
	Mat result, tempImage;
	double min, max;
	Point pMin, pMax, p3;
	matchTemplate(*image, *temp, result, CV_TM_SQDIFF_NORMED);
	minMaxLoc(result, &min, &max, &pMin, &pMax, Mat());
	cout << "Minimum Value" << min << endl;
	cout << "Maximum Value" << max << endl;
	p3.x = pMin.x + temp->cols;
	p3.y = pMin.y + temp->rows;
	tempImage = *image;
	if (min < 0.1) 
	{ //threshold for minimum the smaller the value the better the confidence level for SQDIFF
		Rect crop(pMin, p3);
		Mat updateTemplate;
		updateTemplate = tempImage(crop);
		updateTemplate.copyTo(*temp);
		Rect rec = Rect(pMin.x, pMin.y, temp->cols, temp->rows);
		rectangle(tempImage, rec, Scalar(0, 0, 255), 2, 8, 0);
		imshow("Current Template", *temp);
	}
	auto createCircle = [image](const Point& n) {circle(*image, n, 5, Scalar(0, 0, 255),5,8,0); };
	for_each(AOI.begin(), AOI.end(), createCircle);
	if (AOI.size() == AOIPointsNum) {
		getTemplate(*image);
	}
	imshow("Template Matching", tempImage);
}

void edgeDetector(Mat image) {
	Mat edges;
	cvtColor(image, edges, CV_BGR2GRAY);
	GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
	Canny(edges, edges, 0, 30, 3);
	imshow("edges", edges);
}

int main (int,char**) {
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	#ifdef EDGE
	namedWindow("edges", 1);
	#endif
	Mat globalImage;
	namedWindow("Template Matching");
	setMouseCallback("Template Matching", callBackFunction, NULL);
	Mat training_template;
	training_template = imread("C:\\Users\\faraz.bhatti\\Documents\\Visual Studio 2015\\Projects\\OpenCVIntro\\template.png");
	for (;;)
	{
		cap >> globalImage;
		templateMatcher(&globalImage, &training_template);
		if (ready) {
			training_template = imread("C:\\Users\\faraz.bhatti\\Documents\\Visual Studio 2015\\Projects\\OpenCVIntro\\template.png");
			ready = false;
		}
		#ifdef EDGE
		edgeDetector(globalImage);
		#endif
		if (waitKey(10) == 27)break;
		
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}