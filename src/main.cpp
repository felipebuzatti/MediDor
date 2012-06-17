#include <deque>
#include <string>
#include <sstream>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

using namespace std;
using namespace cv;

// ============================================== //

const int MAX_FRAMES = 50;
typedef enum {WAITING, RECORDING, SELECTING, SHOWING} State;
void onMouse( int event, int x, int y, int, void* );
void processFrame(Mat frame);
void showInstruction(string s);
double getTime();
string convertDouble(double number);
string convertInt(int number);
void setUp();
void help();

// ============================================== //

State currentState;
string mainWindow;
VideoCapture capture;
Mat imgFrame;
Mat measurements;
vector<Mat> capturedImages;
Mat firstImg;
Point p1, p2, p3, p4;
int pointNo;
int windowWidth, windowHeight;
double lastTime;
bool finished;
bool hasOpticalFlow;
char key;

// ============================================== //

int main(int argc, const char *argv[])
{
	setUp();

    while ( !finished )
    {
        Mat frame;
        capture >> frame;
        processFrame(frame);
        imshow(mainWindow, imgFrame);
    }
    
    return 0;
}

// ============================================== //

void help()
{
	cout << "Structure From Linear Horizontal Motion" << endl;
	cout << "  Instructions: " << endl;
	cout << "     Press Q or ESC to quit." << endl;
}

// ============================================== //

void setUp()
{
	// Show help and instructions
	help();

	// Initialize some variables
	currentState = WAITING;
	lastTime = getTime();
	key = -1;
	pointNo = -1;
	finished = false;
	hasOpticalFlow = false;

	// Create main window
	mainWindow = "TP de Visão";
	windowWidth = 640;
	windowHeight = 480;
	imgFrame = Mat::zeros(windowHeight, windowWidth, CV_32FC4);
	namedWindow(mainWindow);
	setMouseCallback( mainWindow, onMouse, 0 );

	// Open video feed
    //capture = VideoCapture("http://192.168.1.104:8080/videofeed");
    capture = VideoCapture(0);
    if (!capture.isOpened()) exit(1);
}

// ============================================== //

void processFrame(Mat frame) {

	frame.copyTo(imgFrame);

	if (currentState == State::WAITING) {
		showInstruction("Press S to start.");
	}
	else if (currentState == State::RECORDING) {

		if (capturedImages.empty()) {
			frame.copyTo(firstImg);
		}
		else if (capturedImages.size() >= MAX_FRAMES-1) {
			currentState = State::SELECTING;
			pointNo = 0;
		}

		// Copy current frame to the buffer
		Mat currFrame = Mat();
		//frame.copyTo(currFrame);
		cvtColor(frame, currFrame, COLOR_BGR2GRAY, CV_8U);
		capturedImages.push_back(currFrame);
		string text = "Captured: " + convertInt(capturedImages.size()) + "/" + convertInt(MAX_FRAMES);
		showInstruction(text);
		/*
		if (getTime() - lastTime > 0.1) {
			lastTime = getTime();
			cout << lastTime << endl;
		}
		*/
	}
	else if (currentState == State::SELECTING) {

		// Show first image
		firstImg.copyTo(imgFrame);

		/*
		Mat grayImg = Mat();
		cvtColor(imgFrame, grayImg, COLOR_BGR2GRAY );
		Ptr<FeatureDetector> fd = FeatureDetector::create("FAST");
		vector<vector<KeyPoint>> keypoints;
		fd->detect(capturedImages, keypoints);
		*/

		// Select and show points
		if (pointNo <= 3) {
			showInstruction("Select the points.");
			if (pointNo > 3) circle(imgFrame, p4, 3, Scalar(0, 200, 0));
			if (pointNo > 2) circle(imgFrame, p3, 3, Scalar(0, 200, 0));
			if (pointNo > 1) circle(imgFrame, p2, 3, Scalar(0, 200, 0));
			if (pointNo > 0) circle(imgFrame, p1, 3, Scalar(0, 200, 0));
		}
		// Confirm point selection
		else if (pointNo == 4) {
			circle(imgFrame, p4, 3, Scalar(0, 200, 0));
			circle(imgFrame, p3, 3, Scalar(0, 200, 0));
			circle(imgFrame, p2, 3, Scalar(0, 200, 0));
			circle(imgFrame, p1, 3, Scalar(0, 200, 0));
			showInstruction("Press N to continue or P to reselect points.");
		}
		else {
			showInstruction("Computing Optical flow.");
			imshow(mainWindow, imgFrame);

			// Compute the measurements
			vector<Point> prevPoints;
			for (int i = 0; i < capturedImages.size(); ++i) {
				if (i == 0) {
					int points[] = {p1.x, p1.y,  p2.x, p2.y,  p3.x, p3.y,  p4.x, p4.y};
					measurements = Mat(8, MAX_FRAMES, CV_32F);
					measurements.col(0) = Mat(8, 1, CV_32F, points);
					prevPoints.push_back(p1);
					prevPoints.push_back(p2);
					prevPoints.push_back(p3);
					prevPoints.push_back(p4);
				}
				else {
					Mat statusMat = Mat();
					Mat errMat = Mat();
					vector<Point> newPoints;
					calcOpticalFlowPyrLK(capturedImages[i-1], capturedImages[i], prevPoints, newPoints, statusMat, errMat);
					if( sum(statusMat)[0] == prevPoints.size() ) {
						int points[] = {newPoints[0].x, newPoints[0].y,  newPoints[1].x, newPoints[1].y,  newPoints[2].x, newPoints[2].y,  newPoints[3].x, newPoints[3].y};
						prevPoints = newPoints;
					}
				}
			}

			currentState = State::SHOWING;
		}
		//showInstruction("Press N to continue.");

	}
	else if (currentState == State::SHOWING) {
		capturedImages[MAX_FRAMES/2].copyTo(imgFrame);
	}

    key = waitKey(50);
    if ( key == 27 || key == 'q' ) finished = true;
    else if (key == 's' && currentState == State::WAITING) {
    	currentState = State::RECORDING;
    	capturedImages.clear();
    	lastTime = getTime();
    }
    else if (key == 'n' && currentState == State::SELECTING && pointNo == 4) {
    	pointNo++;
    }
    else if (key == 'p' && currentState == State::SELECTING && pointNo == 4) {
    	pointNo = 0;
    }
}

// ============================================== //

void showInstruction(string s) {
	int baseLine = 0;
	Size textSize = getTextSize(s, 1, 1, 1, &baseLine );
	Point textOrigin = Point(windowWidth/2 - textSize.width/2, windowHeight - 2*baseLine - 10);
	putText(imgFrame, s, textOrigin, 1, 1, Scalar(255,255,255), 1);
}

// ============================================== //

double getTime() {
	return getTickCount() / getTickFrequency();
}

// ============================================== //

void onMouse( int event, int x, int y, int, void* )
{
	if (event != CV_EVENT_LBUTTONDOWN) return;

	cout << "Mouse position: (" << x << ", " << y << ")" << endl;

	switch (pointNo) {
		case 0:
			p1 = Point(x, y);
			pointNo++;
			break;
		case 1:
			p2 = Point(x, y);
			pointNo++;
			break;
		case 2:
			p3 = Point(x, y);
			pointNo++;
			break;
		case 3:
			p4 = Point(x, y);
			pointNo++;
			break;
		default:
			break;
	}

	/*
    if( event != CV_EVENT_LBUTTONDOWN )
        return;

    Point seed = Point(x,y);
    int lo = ffillMode == 0 ? 0 : loDiff;
    int up = ffillMode == 0 ? 0 : upDiff;
    int flags = connectivity + (newMaskVal << 8) +
                (ffillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
    int b = (unsigned)theRNG() & 255;
    int g = (unsigned)theRNG() & 255;
    int r = (unsigned)theRNG() & 255;
    Rect ccomp;

    Scalar newVal = isColor ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);
    Mat dst = isColor ? image : gray;
    int area;

    if( useMask )
    {
        threshold(mask, mask, 1, 128, CV_THRESH_BINARY);
        area = floodFill(dst, mask, seed, newVal, &ccomp, Scalar(lo, lo, lo),
                  Scalar(up, up, up), flags);
        imshow( "mask", mask );
    }
    else
    {
        area = floodFill(dst, seed, newVal, &ccomp, Scalar(lo, lo, lo),
                  Scalar(up, up, up), flags);
    }

    imshow("image", dst);
    cout << area << " pixels were repainted\n";
    */
}

// ============================================== //

string convertDouble(double number)
{
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

// ============================================== //

string convertInt(int number)
{
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

// ============================================== //
