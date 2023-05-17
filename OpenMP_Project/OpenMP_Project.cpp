#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    double start_time, end_time;  //variables to calculate time of each method
    //Upload image using CV Mat class
    cv::Mat image = cv::imread("..\\Lena.png", cv::IMREAD_COLOR);
    //If image is not available, terminate the program
    if (image.empty())
    {
        cout << "Could not find or open the image" << endl;
        return -1;
    }
    String windowTitle = "Original Image"; //Name of the window

    namedWindow(windowTitle); // Create a window

    imshow(windowTitle, image);

    int rows = image.rows;
    int cols = image.cols;
    int kernel_size, numThreads;
    cout << "\n\n\n\n\n\n\nImage rows no: " << rows << "  Image cols no: " << cols;
    cout << "\nEnter the kernel size (odd value): ";
    cin >> kernel_size;
    //If the kernel size is even, add 1 to be odd
    if (kernel_size % 2 == 0)
        kernel_size += 1;
    int border_size = kernel_size / 2;

    std::cout << "Enter the number of threads: ";
    cin >> numThreads;
    omp_set_num_threads(numThreads);
    start_time = omp_get_wtime(); //Get start time of the parallel section

    //Parallelize the outer loop, allowing multiple threads to process different rows of the image concurrently.
    //Iterate over the pixels (rows and cols) of the image 
#pragma omp parallel num_threads(numThreads) 
    #pragma omp for collapse(2) 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);

          // Perform low-pass filtering operation on the pixel with the Kernel size the user entered.
          // Iterates over a kernel_sizexkernel_size kernel centered around the current pixel as the origin.
            int blueSum = 0, greenSum = 0, redSum = 0, count = 0;

            #pragma omp parallel for collapse(2) reduction(+:greenSum, redSum, blueSum, count) 
            for (int k = -border_size; k <= border_size; k++) {
                for (int l = -border_size; l <= border_size; l++) {

                    int kernel_row = i + k;
                    int kernel_col = j + l;

                    // Check if the current neighbor pixel is within the image 
                    if (kernel_row >= 0 && kernel_row < rows && kernel_col >= 0 && kernel_col < cols) {
                        blueSum += image.at<cv::Vec3b>(kernel_row, kernel_col)[0];
                        greenSum += image.at<cv::Vec3b>(kernel_row, kernel_col)[1];
                        redSum += image.at<cv::Vec3b>(kernel_row, kernel_col)[2];
                        count++;
                    }
                }
            }
            //Calculate the average value of the 3 channels in the current pixel
            pixel[0] = blueSum / count;  // Blue channel
            pixel[1] = greenSum / count;  // Green channel
            pixel[2] = redSum / count;  // Red channel
        }
    }

    end_time = omp_get_wtime(); //Get end time of the parallel section
    cv::imshow("Blurred Image", image);

    std::cout << "Time elapsed: " << end_time - start_time << " seconds\n";

    //Wait for any keystroke in the window
    waitKey(0);
    destroyAllWindows();

    return 0;
    
}
