#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/video/tracking.hpp>
#include <ctime>
#include "Edge_Detection_SaiPrajwal.h"

using namespace cv::xfeatures2d;
using namespace cv;
using namespace std;

static void help()
{
	// print a welcome message, and the OpenCV version
	cout << "\nThis is a demo of Lukas-Kanade optical flow using BRISK and ORB,\n"
		"Using OpenCV version " << CV_VERSION << endl;
	cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n" << endl;
}

string IntToStr(int n)
{
	stringstream result;
	result << n;
	return result.str();
}

Point2f point;
bool addRemovePt = false;

int main(int argc, char** argv)
{
	int dWidth1, dHeight1, count = 0;
	string filename;
	VideoCapture cap;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);


	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;

	cv::CommandLineParser parser(argc, argv, "{@input||}{help h||}");
	string input = parser.get<string>("@input"); // Enter the filepath for video here if you like to analyze OptFlow for video.

	help();

	if (input.empty())
		cap.open(0); //Opens up the default camera
	// In the above line, replace argument with the file path/file name if you have a video
	// For example, cap.open("C:/Users/Prajwal/Desktop/SampleVideo.mp4");
	else if (input.size() == 1 && isdigit(input[0]))
		cap.open(input[0] - '0');
	else
		cap.open(input);

	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}

	namedWindow("Opt Flow", CV_WINDOW_AUTOSIZE); //create a window called "Opt Flow"
	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

	// The below line saves the video on to my Desktop. Make sure you change the path before execution 
	VideoWriter oVideoWriter("C:/Users/Prajwal/Desktop/MotionEstimation_partB.avi", CV_FOURCC('P', 'I', 'M', '1'), 20, frameSize, true);
	
	if (!oVideoWriter.isOpened()) //if not initialize the VideoWriter successfully, exit the program
	{
		cout << "ERROR: Failed to write the video" << endl;
		return -1;
	}

	Mat gray, prevGray, image, frame, dummy;
	vector<Point2f> points[2];

	for (;;)
	{
		clock_t begin = clock(); //This is used to find the execution time taken for the program
		cap >> frame;
		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);

		if (needToInit) // This condition gets satisfied once we press 'r' on keyboard
		{
			// automatic initialization

			// For BRISK to detect keypoints, keep the following line.
			//Ptr<Feature2D> f2d = BRISK::create(50, 4, 1.0f);

			// For ORB to detect Keypoints, keep the following line.
			Ptr<Feature2D> f2d = ORB::create(100, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 80);

			Mat descriptors; std::vector<KeyPoint> keypoints;
			f2d->detect(gray, keypoints);
			KeyPoint::convert(keypoints, points[1]);

			image.copyTo(dummy);
			dummy = dummy - image; // For frame 1 of auto-initialization, dummy will be a black image of size that is same as 'image'.

			addRemovePt = false;
		}
		else if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001); //Calculating the optical flow. We get points[1] which match the ones in previous frame
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				if (addRemovePt)
				{
					if (norm(point - points[1][i]) <= 5)
					{
						addRemovePt = false;
						continue;
					}
				}

				if (!status[i])
					continue;
				/* For every point, assign a different color. Here, I took 4 colors- Green, Red, Blue and White */
				if (i % 4 == 0)
					line(dummy, points[0][i], points[1][i], Scalar(0, 255, 0), 2, -1);
				else if (i % 4 == 1)
					line(dummy, points[0][i], points[1][i], Scalar(0, 0, 255), 2, -1);
				else if (i % 4 == 2)
					line(dummy, points[0][i], points[1][i], Scalar(255, 0, 0), 2, -1);
				else
					line(dummy, points[0][i], points[1][i], Scalar(255, 255, 255), 2, -1);
				image = (image + dummy);
				points[1][k++] = points[1][i];

			}
			points[1].resize(k);
			oVideoWriter.write(image); //Writes the frames into the output video file
		}

		if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
		{
			vector<Point2f> tmp;
			tmp.push_back(point);
			cornerSubPix(gray, tmp, winSize, Size(-1, -1), termcrit);
			points[1].push_back(tmp[0]);
			addRemovePt = false;
		}

		needToInit = false;
		cv::imshow("Opt Flow", image);

		char c = (char)waitKey(10);
		if (c == 27) //Corresponds to escape
			break;
		switch (c)
		{
		case 'r':
			needToInit = true;
			break;
		case 'c':
			points[0].clear();
			points[1].clear();
			break;
		}

		std::swap(points[1], points[0]); // Previous points = Current points
		cv::swap(prevGray, gray); // Previous grayscale image = Current gray scale image
		clock_t end = clock(); //Ends time
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		cout << "It took " << elapsed_secs << " second(s).\n" << endl; //Calculates the time taken to run one loop
	}

	return 0;
}