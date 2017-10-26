/*
Door is:
-moving
-square
-taking up most of screen

We can identify it by its shape and size, and the fact that it is moving.
Colour identification is useless because of the changes in lighting conditions etc.

APPROACH IDEA
- GMM - gives a binary image of the moving pixels (moving pixels are white)
- Canny edge detection - outputs a binary image
- Probabilistic Hough line segments  on the edge image - get the longest lines, and convert this image to binary (door edge lines are white)
- Now get the moving lines by ANDing the Hough Lines image with the movement image - gives moving long lines corresponding to the door

*/




#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/features2d.hpp"
#include <stdio.h>
#include "Utilities.h"
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;


#define VIDEO_1_INDEX 0
#define VIDEO_2_INDEX 1
#define VIDEO_3_INDEX 2
#define VIDEO_4_INDEX 3


int TN = 0, FN = 0, TP = 0, FP = 0;						//false/true positives/negatives for metrics
int num_hough_pixels;									//number of pixels in image that correspond to hough lines
//the ground-truth open/close times for each video
int groundtruthtimes_video1[] = {  94, 211, 373, 490 },  
	groundtruthtimes_video2[] = {  71, 184, 339, 460 },
	groundtruthtimes_video3[] = { 153, 251, 380, 492 }, 
	groundtruthtimes_video4[] = { 134, 230, 311, 432 };
int groundtruthframes[] = { 563,533,613,567 };			//the ground-truth number of frames per video
int top_left_coordinates[] = { 76, 65, 221, 264 };		//top-left x, y coordinates for video 1and 2(77, 65) and videos 3 and 4 (221,264)
int bottom_right_coordinates[] = { 225,359,573,1040 };
Mat gndtruth_1and2, gndtruth_3and4;//these are the ground truth images for the door location
Mat video_first_frame[4];								//stores the first frame of each video
Mat prev_frame;
int thresh[4] = { 100,100,100,100};
bool door_detected = false;
//groundtruth for each frame, i.e. whether door moving in each frame or not
int video_1_door_frames_groundtruth[563], video_2_door_frames_groundtruth[533],
	video_3_door_frames_groundtruth[613], video_4_door_frames_groundtruth[567];
//record of door detection for each frame, i.e. whether door detected in frame or not
int video_1_door_frames_detected[563], video_2_door_frames_detected[533],
	video_3_door_frames_detected[613], video_4_door_frames_detected[567];

/*				*
 *	FUNCTIONS	*
 *				*/

void computeGndTruthFrames();																	//Fills the gnd truth frames arrays
void computeGndTruthLocation(Mat outputimg, int bl_x, int tl_y, int tr_x, int br_y);			//Creates a ground truth location for the door in each video
void processVideo(VideoCapture& video, int starting_frame_number, int end, int video_index);	//analyse the videos
void checkIfDoorDetected(int video_num, int num_frame, Mat& resulting_image, Mat& ground_truth_location_image);
void storeDoorDetectedState(int video_number, int num_frame);									//stores the state of detection of the door for a frame
void computeMetrics(int video_number);															//computes metrics (TP, FP, TN, FN)
void printMetrics(int video_number);															//prints metrics



int main(int argc, const char** argv) {


	char* file_location = "Media/";
	// Load video(s)
	char* video_files[] = { "Door1.avi", "Door2.avi", "Door3.avi", "Door4.avi" };
	int number_of_videos = sizeof(video_files) / sizeof(video_files[0]);
	VideoCapture* video = new VideoCapture[number_of_videos];
	for (int video_file_no = 0; (video_file_no < number_of_videos); video_file_no++)
	{
		string filename(file_location);
		filename.append(video_files[video_file_no]);
		video[video_file_no].open(filename);
		if (!video[video_file_no].isOpened())
		{
			cout << "Cannot open video file: " << filename << endl;
			return -1;
		}
	}
	//create the ground truth information for the frames in which the door should be detected
	computeGndTruthFrames();
	//process each of the 4 videos, and compute their metrics
	processVideo(video[VIDEO_1_INDEX], 0, 563, VIDEO_1_INDEX);
	computeMetrics(VIDEO_1_INDEX);
	processVideo(video[VIDEO_2_INDEX], 0, 533, VIDEO_2_INDEX);
	computeMetrics(VIDEO_2_INDEX);
	processVideo(video[VIDEO_3_INDEX], 0, 613, VIDEO_3_INDEX);
	computeMetrics(VIDEO_3_INDEX);
	processVideo(video[VIDEO_4_INDEX], 0, 567, VIDEO_4_INDEX);
	computeMetrics(VIDEO_4_INDEX);


	waitKey(0);
	return 0;
}


//compute the number of true/false negatives/positives
void computeMetrics(int video_number) {
	TP = 0, FP = 0, TN = 0, FP = 0; //reset
	
	if (video_number == 0) {
		for (int i = 0; i < groundtruthframes[0]; i++) {
			if (video_1_door_frames_groundtruth[i] == true && video_1_door_frames_detected[i] == true)//detected is true
				TP++;
			else if (video_1_door_frames_groundtruth[i] == true && video_1_door_frames_detected[i] == false)//not detected is false
				FN++;
			else if (video_1_door_frames_groundtruth[i] == false && video_1_door_frames_detected[i] == false)//not detected is true
				TN++;
			else if (video_1_door_frames_groundtruth[i] == false && video_1_door_frames_detected[i] == true)//not detected is false
				FP++;
		}
	}
	else if (video_number == 1) {
		for (int i = 0; i < groundtruthframes[1]; i++) {
			if (video_2_door_frames_groundtruth[i] == true && video_2_door_frames_detected[i] == true)
				TP++;
			else if (video_2_door_frames_groundtruth[i] == true && video_2_door_frames_detected[i] == false)
				FN++;
			else if (video_2_door_frames_groundtruth[i] == false && video_2_door_frames_detected[i] == false)
				TN++;
			else if (video_2_door_frames_groundtruth[i] == false && video_2_door_frames_detected[i] == true)
				FP++;
		}
	}
	else if (video_number == 2) {
		for (int i = 0; i < groundtruthframes[2]; i++) {
			if (video_3_door_frames_groundtruth[i] == true && video_3_door_frames_detected[i] == true)
				TP++;
			else if (video_3_door_frames_groundtruth[i] == true && video_3_door_frames_detected[i] == false)
				FN++;
			else if (video_3_door_frames_groundtruth[i] == false && video_3_door_frames_detected[i] == false)
				TN++;
			else if (video_3_door_frames_groundtruth[i] == false && video_3_door_frames_detected[i] == true)
				FP++;
		}
	}
	else {//video 4, i.e. video_number = 3
		for (int i = 0; i < groundtruthframes[3]; i++) {
			if (video_4_door_frames_groundtruth[i] == true && video_4_door_frames_detected[i] == true)
				TP++;
			else if (video_4_door_frames_groundtruth[i] == true && video_4_door_frames_detected[i] == false)
				FN++;
			else if (video_4_door_frames_groundtruth[i] == false && video_4_door_frames_detected[i] == false)
				TN++;
			else if (video_4_door_frames_groundtruth[i] == false && video_4_door_frames_detected[i] == true)
				FP++;
		}
	}
	//output the metrics to the console window
	printMetrics(video_number);
}


//prints the metrics previously computed for a given video
void printMetrics(int video_number) {
	//calculate performance metrics
	double precision = ((double)TP) / ((double)(TP + FP));
	double recall = ((double)TP) / ((double)(TP + FN));
	double accuracy = ((double)(TP + TN)) / ((double)(TP + FP + TN + FN));
	double specificity = ((double)TN) / ((double)(FP + TN));
	double f1 = 2.0*precision*recall / (precision + recall);
	//output results to console
	cout << "-----------------------------\nMetrics for Video " << video_number << "\n-----------------------------" << endl;
	cout << "True postitives: " << TP << endl;
	cout << "False postitives: " << FP << endl;
	cout << "True negatives: " << TN<< endl;
	cout << "False negatives: " << FN << endl;
	cout << "-----------------------------" << endl;
	cout << "Precision: " << precision * 100 << endl;
	cout << "Recall: " << recall * 100 << endl;
	cout << "Accuracy: " << accuracy * 100 << endl;
	cout << "Specificity: " << specificity * 100 << endl;
	cout << "f1: " << f1 * 100 << endl;
	cout << "-----------------------------" << endl << endl;
}


//Used to set all of the Hough line pixels to white, and all other pixels to black, i.e. binarising Hough Lines image
Mat setEdgePixelsToWhite(Mat& image) {
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			if (image.at<char>(row, col) == 76) {	//where 76 is the greyscale value of the hough lines
				image.at<char>(row, col) = 255;		//set to white
			}
			else image.at<char>(row, col) = 0;		//else set to black
		}
	}
	return image;
}


//gets the number of pixels in the binary image that belong to the object(Assumed white)
int getNumObjectPixels(Mat& img)
{
	int count = 0;

	for (int rows = 0; rows < img.rows; rows++)
		for (int cols = 0; cols < img.cols; cols++)
			if (img.at<char>(rows, cols) != 0)
				count++;
	//this is the number of object pixels
	return count;
}


//create a ground truth image, isolating the door from the rest of the image
void computeGndTruthLocation(Mat outputimg, int bl_x, int tl_y, int tr_x, int br_y) {
	for (int row = 0; row < outputimg.rows; row++) {
		for (int col = 0; col < outputimg.cols; col++) {
			if ((col >= bl_x && col <= tr_x)&&  (row >= tl_y && row <= br_y)) {//if we are between the lhs and rhs of the door
					outputimg.at<char>(row, col) = 255;//white
				}
			else outputimg.at<char>(row, col) = 0;//black
		}
	}
}


//fills the arrays with the correct answers for whether a door should be detected or not
void computeGndTruthFrames(){
	
	int i;
	//first initialise all of the ground truth times to be 0, and add in the frames where the door is being opened or closed after
	for (i = 0; i < groundtruthframes[0]; i++) video_1_door_frames_groundtruth[i] = 0;
	for (i = 0; i < groundtruthframes[1]; i++) video_2_door_frames_groundtruth[i] = 0;
	for (i = 0; i < groundtruthframes[2]; i++) video_3_door_frames_groundtruth[i] = 0;
	for (i = 0; i < groundtruthframes[3]; i++) video_4_door_frames_groundtruth[i] = 0;

	//video 1 - first open/close
	for (i = groundtruthtimes_video1[0]; i < groundtruthtimes_video1[1]; i++) {
		video_1_door_frames_groundtruth[i] = 1;
	}
	//video1 - second open/close
	for (i = groundtruthtimes_video1[2]; i < groundtruthtimes_video1[3]; i++) {
		video_1_door_frames_groundtruth[i] = 1;
	}

	//video 2 - first open/close
	for (i = groundtruthtimes_video2[0]; i < groundtruthtimes_video2[1]; i++) {
		video_2_door_frames_groundtruth[i] = 1;
	}
	//video2  - second open/close
	for (i = groundtruthtimes_video2[2]; i < groundtruthtimes_video2[3]; i++) {
		video_2_door_frames_groundtruth[i] = 1;
	}

	//video 3  - first open/close
	for (i = groundtruthtimes_video3[0]; i < groundtruthtimes_video3[1]; i++) {
		video_3_door_frames_groundtruth[i] = 1;
	}
	//video3 - second open/close
	for (i = groundtruthtimes_video3[2]; i < groundtruthtimes_video3[3]; i++) {
		video_3_door_frames_groundtruth[i] = 1;
	}

	//video 4  - first open/close
	for (i = groundtruthtimes_video4[0]; i < groundtruthtimes_video1[1]; i++) {
		video_1_door_frames_groundtruth[i] = 1;
	}
	//video 4 - second open/close
	for (i = groundtruthtimes_video1[2]; i < groundtruthtimes_video1[3]; i++) {
		video_1_door_frames_groundtruth[i] = 1;
	}
}


//function to check whether we have detected a door, and update the global arrays for the specific frames if we have
void storeDoorDetectedState(int video_number, int num_frame){// , Mat& door_image) {
	if (door_detected == true) {			//set the corresponding frame to true
		if (video_number == 0)
			video_1_door_frames_detected[num_frame] = 1;

		else if (video_number == 1) {
			video_2_door_frames_detected[num_frame] = 1;
		}
		else if (video_number == 2) {
			video_3_door_frames_detected[num_frame] = 1;
		}
		else if (video_number == 3) {
			video_4_door_frames_detected[num_frame] = 1;
		}
	}
	else if (door_detected == false) {		//set the corresponding frame to false
		if (video_number == 0)
			video_1_door_frames_detected[num_frame] = 0;

		else if (video_number == 1) {
			video_2_door_frames_detected[num_frame] = 0;
		}
		else if (video_number == 2) {
			video_3_door_frames_detected[num_frame] = 0;
		}
		else if (video_number == 3) {
			video_4_door_frames_detected[num_frame] = 0;
		}
	}
}


//validates the image to see whether a door has been detected by the algorithm
void checkIfDoorDetected(int video_num, int num_frame, Mat& resulting_image, Mat& ground_truth_location_image) {
	/*
	Door detected if the white pixels corresponding to the door are in the region of interest
	Assume that we have done connected components and have been able to identify the door region
	If we have the door region, it will take up most of this ROI
	so if we have a counter for the number of white pixels in this region, if there are more white than black, then we will have a door detected
	*/

	//mask out the background
	bitwise_and(resulting_image, ground_truth_location_image, resulting_image);
	imshow("resulting image", resulting_image);

	int num_object_pixels = getNumObjectPixels(resulting_image);
	if (num_object_pixels>=thresh[video_num])
		door_detected=true;
	else door_detected = false;
	//store the answer for this frame in the global array for comparison to ground truth later
	storeDoorDetectedState(video_num, num_frame);
}


//process the videos frame by frame, looking for a door
void processVideo(VideoCapture& video, int starting_frame, int end, int video_index) {

	Mat current_frame, grey_frame, hough_lines_image, grey_canny, prev_frame;

	//capture the video
	video.set(CV_CAP_PROP_POS_FRAMES, 0);
	video.read(current_frame);
	video_first_frame[video_index] = current_frame;
	//Construct the door location gndtruths now that we have the initial frame of the video
	cvtColor(current_frame, grey_frame, CV_BGR2GRAY);
	if (video_index == 0){//same for videos 1 and 2
		gndtruth_1and2 = grey_frame.clone();
		computeGndTruthLocation(gndtruth_1and2,  top_left_coordinates[0], top_left_coordinates[1], bottom_right_coordinates[0], bottom_right_coordinates[1]);
		imshow("Groundtruth location of door (videos 1 and 2)",gndtruth_1and2);
	}
	if (video_index == 2){//same for videos 3 and 4
		gndtruth_3and4 = grey_frame.clone();
		computeGndTruthLocation(gndtruth_3and4, top_left_coordinates[2], top_left_coordinates[3], bottom_right_coordinates[2], bottom_right_coordinates[3]);
		imshow("Groundtruth location of door (videos 3 and 4)", gndtruth_3and4);
	}
	

	//get video frame information
	int number_of_frames = video.get(CAP_PROP_FRAME_COUNT);
	cout << "Number of frames: " << number_of_frames << endl;
	double frame_rate = video.get(CV_CAP_PROP_FPS);//get the frames per second
	double time_between_frames = 1000.0 / frame_rate;
	

	//set up GMM variables
	Ptr<BackgroundSubtractorMOG2> gmm = createBackgroundSubtractorMOG2();
	Mat foreground_mask, foreground_image = Mat::zeros(current_frame.size(), CV_8UC3), thresholded_image;
	Mat structuring_element(3, 3, CV_8U, Scalar(1)), closed_image;
	
	
	//go through the frames to the end of the video
	int frame_count = 0;
	while ((!current_frame.empty()) && (frame_count, number_of_frames)) {
		door_detected = false;//reset
		
		//first detect the edges in the image using Canny edge detection
		cvtColor(current_frame, grey_frame, CV_BGR2GRAY);
		blur(grey_frame, grey_canny, Size(3, 3));
		if (video_index == 0 || video_index == 1)			//because this frame size is smaller
			Canny(grey_frame, grey_canny, THRESH_OTSU, 255, 3);
		else Canny(grey_frame, grey_canny, THRESH_OTSU, 300, 3);	//because this frame size is larger	
		imshow("Canny image", grey_canny);

		//dilate and close the edge image to get more consistent lines
		dilate(grey_canny, grey_canny, Mat());
		//morphologyEx(grey_canny, grey_canny, MORPH_CLOSE, Mat());
		//morphologyEx(grey_canny, grey_canny, MORPH_CLOSE, Mat());
		imshow("dilated and closed canny", grey_canny);

		//Probabilistic Hough line segments - extract the longest lines from the image (door)
		vector<Vec4i> lines(600);
		cvtColor(grey_canny, hough_lines_image, CV_GRAY2BGR);	//works on BGR image
		if (video_index == 0 || video_index == 1)				//use smaller line-length threshold because of smaller image size
			HoughLinesP(grey_canny, lines, 1, CV_PI / 180, THRESH_OTSU, 170, 5);
		else 													//use larger line-length threshold because of larger image size
			HoughLinesP(grey_canny, lines, 1, CV_PI / 180, THRESH_OTSU, 400, 5);
		//draw the hough lines
		for (size_t i = 0; i < lines.size(); i++){
			line(hough_lines_image, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
		}
		imshow("Hough lines image", hough_lines_image);
		//convert hough image to greyscale and then make binary
		Mat grey_hough_lines_image, binary_hough;
		cvtColor(hough_lines_image, grey_hough_lines_image, CV_BGR2GRAY);
		grey_hough_lines_image=setEdgePixelsToWhite(grey_hough_lines_image);
		num_hough_pixels = getNumObjectPixels(grey_hough_lines_image);
		imshow("now look!", grey_hough_lines_image);
		threshold(grey_hough_lines_image, grey_hough_lines_image, THRESH_OTSU, 255, THRESH_BINARY);
		

		//GAUSSIAN MIXTURE MODEL - identify the moving pixels in the image
		gmm->apply(current_frame, foreground_mask);
		threshold(foreground_mask, thresholded_image, THRESH_OTSU, 255, THRESH_BINARY);
		Mat moving_incl_shadows, shadow_points;
		threshold(foreground_mask, moving_incl_shadows, THRESH_OTSU, 255, THRESH_BINARY);
		imshow("Thresholded image", thresholded_image);
		absdiff( thresholded_image, moving_incl_shadows, shadow_points );	
		imshow("Shadow points", shadow_points);
		Mat cleaned_foreground_mask;
		morphologyEx(thresholded_image, closed_image, MORPH_CLOSE, structuring_element);
		morphologyEx(closed_image, cleaned_foreground_mask, MORPH_OPEN, structuring_element);//had this commented out??
		foreground_image.setTo(Scalar(0, 0, 0));
		current_frame.copyTo(foreground_image, cleaned_foreground_mask);
		//have a look at the foreground image (moving pixels) - want to convert this to bianry
		imshow("foreground image", foreground_image);
		cvtColor(foreground_image, foreground_image, CV_BGR2GRAY);
		//get overall moving pixels in the image and make binary
		bitwise_or(foreground_image, thresholded_image, foreground_image);	
		threshold(foreground_image, foreground_image, THRESH_OTSU, 255, THRESH_BINARY);

		Mat and_result;
		//A door can be identified by the moving Hough Lines - AND the Hough lines with the moving pixels
		bitwise_and(foreground_image, grey_hough_lines_image, and_result);
		imshow("and result", and_result);

		//check if a door was detected - if it was, this function will set the door_detected variable to true
		if(video_index ==0 || video_index==1)
			checkIfDoorDetected(video_index, frame_count, and_result, gndtruth_1and2); 
		else checkIfDoorDetected(video_index, frame_count, and_result, gndtruth_3and4);


		//go to the next frame
		video.read(current_frame);
		frame_count += 1;
		cvWaitKey(time_between_frames / 10);
		prev_frame = current_frame;
	}
}




