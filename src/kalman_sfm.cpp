#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>
using namespace std;

#include "kalman_sfm.hpp"

using namespace cv;

// ============================================== //

double recoverDistance(Mat pixelPositions, Mat cameraMatrix, double baseline) {

    /*
     * Kalman Filter variables
     * State: 4 points, 4 values each (X/Z, Y/Z, Z, V/Z) = total of 16
     * Measures: 4 points, 2 values each (x, y) = total of 8
     * No control
     */
    KalmanFilter KF(16, 8, 0);

	// Set up Transition Matrix and other KF parameters
	KF.transitionMatrix = computeTransitionMatrix(4);
	KF.measurementMatrix = computeMeasurementMatrix(4, cameraMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e0));
	setIdentity(KF.errorCovPost, Scalar::all(1));
	randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));

	// Run the filter for the n measured positions
	for (int i = 0; i < pixelPositions.cols; ++i) {

		// Compute the predicted state
		KF.predict();

		// Correct the Kalman filter based on the measurement
	    Mat measurement = getMeasurement(pixelPositions, i);
	    adjustMeasurement(measurement, cameraMatrix);
		KF.correct(measurement);
	}

	// Compute the final prediction
	Mat prediction = KF.predict();

	// Get the baseline measurement
	double distPredictedBaseline = getDistance(prediction, 0, 1);
	double distPredicted = getDistance(prediction, 2, 3);
	double adjustedDistance = distPredicted * baseline/distPredictedBaseline;

	cout << "Predicted: " << distPredicted << endl;
	cout << "Predicted Baseline: " << distPredictedBaseline << endl;

	return adjustedDistance;
}

// ============================================== //

/**
 * Compute the Kalman Filter Transition Matrix, which maps one state to
 * the next. The base assumption here is that we have only horizontal
 * movement, so the points position, in a camera-centric world, changes
 * just on the X coordinate.
 *
 * @param numPoints the amount of points to recover
 * @return the Transition Matrix
 */
Mat computeTransitionMatrix(int numPoints) {
	Mat transMatrix = Mat::zeros(numPoints*4, numPoints*4, CV_32F);
	for (int i = 0; i < numPoints; ++i) {
		// Transition for X/Z
		transMatrix.at<float>(i*4, i*4 + 0) = 1.0;
		transMatrix.at<float>(i*4, i*4 + 3) = 1.0;

		// Transition for Y/Z
		transMatrix.at<float>(i*4 + 1, i*4 + 1) = 1.0;

		// Transition for Z
		transMatrix.at<float>(i*4 + 2, i*4 + 2) = 1.0;

		// Transition for V/Z
		transMatrix.at<float>(i*4 + 3, i*4 + 3) = 1.0;
	}
	return transMatrix;
}

// ============================================== //

/**
 * Compute the Kalman Filter Measurement Matrix, which maps the current
 * state to the predicted measurement. Here we do a simple perspective
 * projection of the point to find the pixel coordinates (not adjusted
 * by the image center).
 *
 * @param numPoints the amount of points to recover
 * @param cameraMatrx the matrix with the camera intrinsic parameters
 * @return the Measurement Matrix
 */
Mat computeMeasurementMatrix(int numPoints, Mat cameraMatrix) {

	Mat measMatrix = Mat::zeros(numPoints*2, numPoints*4, CV_32F);
	for (int i = 0; i < numPoints; ++i) {
		// Projection for Xim - Cx
		measMatrix.at<float>(i*2, i*4 + 0) = cameraMatrix.at<float>(0, 0);

		// Projection for Yim - Cy
		measMatrix.at<float>(i*2 + 1, i*4 + 1) = cameraMatrix.at<float>(1, 1);
	}
	return measMatrix;
}

// ============================================== //

/**
 * Recover the column in the measured pixel positions which contains
 * the i-th measurement.
 *
 * @param pixlePositions the matrix with all measured pixel positions
 * @param i the measurement number (0 based)
 * @return a Mat header pointing to the right column
 */
Mat inline getMeasurement(Mat pixelPositions, int i) {
	return pixelPositions.col(i);
}

// ============================================== //

/**
 * Adjust the measurement, subtracting from it the camera center,
 * to find the center-based pixel position used as the measurement in
 * the Kalman Filter.
 *
 * @param measurement a Mat (8x1) with the pixel positions of current
 *   measurement.
 * @param cameraMatrix the matrix with camera intrisic parameters
 */
void adjustMeasurement(Mat measurement, Mat cameraMatrix) {
	float Cx = cameraMatrix.at<float>(0,2);
	float Cy = cameraMatrix.at<float>(1,2);
	for (int r = 0; r < measurement.rows/2; ++r) {
		measurement.at<float>(2*r) -= Cx;
		measurement.at<float>(2*r+1) -= Cy;
	}
}

// ============================================== //

/**
 * Compute the distance between two predicted points
 *
 * @param predition the Kalman Filter predicted state
 * @param p1 the first point (0 based)
 * @param p2 the second point (0 based)
 *
 * @return the distance
 */
double getDistance(Mat prediction, int p1, int p2) {
	p1 *= 4;
	p2 *= 4;
	double dX = prediction.at<float>(p1)*prediction.at<float>(p1+2) - prediction.at<float>(p2)*prediction.at<float>(p2+2);
	double dY = prediction.at<float>(p1+1)*prediction.at<float>(p1+2) - prediction.at<float>(p2+1)*prediction.at<float>(p2+2);
	double dZ = prediction.at<float>(p1+2) - prediction.at<float>(p2+2);

	return sqrt(dX*dX + dY*dY + dZ*dZ);
}

// ============================================== //
