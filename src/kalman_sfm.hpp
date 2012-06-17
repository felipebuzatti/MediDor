/*
 * kalman_sfm.hpp
 *
 *  Created on: 17/06/2012
 *      Author: vilela
 */

#ifndef KALMAN_SFM_HPP_
#define KALMAN_SFM_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>

double recoverDistance(cv::Mat pixelPositions, cv::Mat cameraMatrix, double baseline);
cv::Mat computeTransitionMatrix(int numPoints);
cv::Mat computeMeasurementMatrix(int numPoints, cv::Mat cameraMatrix);
cv::Mat inline getMeasurement(cv::Mat pixelPositions, int idx);
void adjustMeasurement(cv::Mat measurement, cv::Mat cameraMatrix);
double getDistance(cv::Mat prediction, int p1, int p2);



#endif /* KALMAN_SFM_HPP_ */
