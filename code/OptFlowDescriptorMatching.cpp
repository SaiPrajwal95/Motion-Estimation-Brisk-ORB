
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/video/tracking.hpp>
#include<ctime>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

void main() {
	//for gray scale images:

	int dWidth1, dHeight1;
	float count;
	string filename;

	VideoCapture cap("C:/Users/Prajwal/Desktop/spl_cut.mp4"); //In place of cap(0), replace with cap("C:/Users/../FileName.mp4") for video file as input

	if (!cap.isOpened())
	{
		cout << "ERROR: Cannot open the video file" << endl;
		return;
	}
	namedWindow("Descriptors Matching", CV_WINDOW_AUTOSIZE);
	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

	// Don't forget to change the file path in the line below.
	VideoWriter oVideoWriter("C:/Users/Prajwal/Desktop/MotionEstimation_partC.avi", CV_FOURCC('P', 'I', 'M', '1'), 20, frameSize, true);
	if (!oVideoWriter.isOpened()) //if not initialize the VideoWriter successfully, exit the program
	{
		cout << "ERROR: Failed to write the video" << endl;
		return;
	}

	Mat img, frame, frame_prev, dummy, gray;

	// For BRISK to detect keypoints, keep the following line.
	Ptr<Feature2D> f2d = BRISK::create(80, 4, 1.0f);

	// For ORB to detect Keypoints, keep the following line.
	//Ptr<Feature2D> f2d = ORB::create(300, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 80);

	gray = img;
	std::vector<KeyPoint> keypoints, keypoints_prev;
	vector<Point2f> pt1, pt2;

	Mat descriptors, descriptors_prev;
	count = 0;

	while (1)
	{
		clock_t begin = clock(); //This is used to find the execution time taken for the program
		++count;
		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "ERROR: Cannot read a frame from video file" << endl;
			break;
		}

		cvtColor(frame, gray, COLOR_BGR2GRAY);

		f2d->detect(gray, keypoints); if (keypoints.empty()) continue; //Detect Keypoints using the method declared above
		f2d->compute(gray, keypoints, descriptors); // Compute the descriptors based on keypoints and the image

													// Convert the type of the descriptors if there aren't in CV_32F already (Because Flann matcher needs them in CV_32F format)  
		if (descriptors.type() != CV_32F) {
			descriptors.convertTo(descriptors, CV_32F);
		}

		if (count == 1) // For the first frame
		{
			descriptors_prev = descriptors; // Make previous descriptors = Present descriptors for 1st frame
			frame.copyTo(frame_prev);
			keypoints_prev = keypoints; // Make previous keypoints = Present keypoints for 1st frame
			frame.copyTo(dummy);
			dummy = dummy - frame; // It's just a black image
		}

		KeyPoint::convert(keypoints_prev, pt1); //Converting Keypoints to Points format to draw lines of Optical Flow
		KeyPoint::convert(keypoints, pt2);

		cv::FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match(descriptors_prev, descriptors, matches); //Computes the matches between previous and present descriptors.

															   // This section computes the minimum and maximum distances
		double max_dist = 0; double min_dist = 100;

		for (int i = 0; i < descriptors_prev.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		//-- Draw only "good" matches (i.e. whose distance is less than 1.75*min_dist,
		//-- or a small arbitary value (125) in the event that min_dist is very small
		//cout << min_dist<< "\n";

		std::vector< DMatch > good_matches;

		for (int i = 0; i < descriptors_prev.rows; i++)
		{
			if (matches[i].distance <= max(2*min_dist, 145.00))
			{
				good_matches.push_back(matches[i]);
			}
		}

		Mat img_matches; //This image will be displayed finally (All the flow lines will be drawn on this)

						 //cout << "\nNo. of matches - " << good_matches.size();
		for (int i = 0; i < (int)good_matches.size(); i++)
		{

			if (i % 4 == 0)
				line(dummy, pt1[good_matches[i].queryIdx], pt2[good_matches[i].trainIdx], Scalar(0, 255, 0), 2, -1);
			else if (i % 4 == 1)
				line(dummy, pt1[good_matches[i].queryIdx], pt2[good_matches[i].trainIdx], Scalar(255, 0, 0), 2, -1);
			else if (i % 4 == 2)
				line(dummy, pt1[good_matches[i].queryIdx], pt2[good_matches[i].trainIdx], Scalar(0, 0, 255), 2, -1);
			else
				line(dummy, pt1[good_matches[i].queryIdx], pt2[good_matches[i].trainIdx], Scalar(255, 255, 255), 2, -1);

			//line(dummy, pt1[good_matches[i].queryIdx], pt2[good_matches[i].trainIdx], Scalar(0, 255, 0), 2, -1);
		}
		frame = frame + dummy;
		cv::imshow("Descriptors Matching", frame);
		oVideoWriter.write(frame); //Writes the frames into the output video file
		int p = waitKey(10);
		if (p == 27)
			break;
		else if (p == 'c') //If we press 'c', the points will be cleared
		{
			count = 0;
		}

		//Copying present values as past values
		std::swap(descriptors_prev, descriptors);
		std::swap(frame_prev, frame);
		std::swap(keypoints_prev, keypoints);
		clock_t end = clock(); //Ends time
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		cout << "It took " << elapsed_secs << " second(s).\n" << endl; //Calculates the time taken to run one loop
	}
}