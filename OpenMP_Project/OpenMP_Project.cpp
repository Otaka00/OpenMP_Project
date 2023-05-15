#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    double start_time, end_time;  //variables to calculate time of each method
    //Upload image using CV Mat class
    cv::Mat image = cv::imread("..\\lena.png", cv::IMREAD_COLOR);
    if (image.empty())
    {
        cout << "Could not find or open the image" << endl;
        return -1;
    }
    String windowTitle = "Normal Image"; //Name of the window

    namedWindow(windowTitle); // Create a window

    imshow(windowTitle, image);

    int rows = image.rows;
    int cols = image.cols;
    int kernel_size = 11, numThreads;
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

    int chunkSize = ((pow(kernel_size, 2)) * rows * cols) / numThreads;
    cout << "\nChunk Size: " << chunkSize;

    //Parallelize the outer loop, allowing multiple threads to process different rows of the image concurrently.
    //Iterate over the pixels (rows and cols) of the image 
#pragma omp parallel for collapse(2) num_threads(numThreads) 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);

            // Perform low-pass filtering operation on the pixel with the Kernel size the user entered.
           // Iterates over a kernel_sizexkernel_size kernel centered around the current pixel
            int blueSum = 0, greenSum = 0, redSum = 0, count = 0;

#pragma omp parallel for collapse(2) num_threads(numThreads) reduction(+:greenSum, redSum, blueSum, count) 
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

/*
cv::Mat performConvolution(const cv::Mat& image, const cv::Mat& kernel) {
    cv::Mat result;
    cv::filter2D(image, result, -1, kernel);
    return result;
}

int main() {
    // Load the image
    cv::Mat image = cv::imread("..\\lena.png", cv::IMREAD_GRAYSCALE);

    // Define the kernel
    cv::Mat kernel = (cv::Mat_<float>(7, 7) << -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, 49, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1);

    // Perform convolution
    cv::Mat convolvedImage = performConvolution(image, kernel);

    // Display the original and convolved images
    cv::imshow("Original Image", image);
    cv::imshow("Convolved Image", convolvedImage);
    cv::waitKey(0);

    return 0;
}

*/



/*
* 
#include <iostream>
#include <vector>
#include <omp.h>



// Function to perform low pass filtering using a 3x3 kernel
void lowPassFilter(const std::vector<std::vector<int>>& input, std::vector<std::vector<int>>& output) {
    int rows = input.size();
    int cols = input[0].size();

#pragma omp parallel for collapse(2)
    for (int i = 2; i < rows - 2; ++i) {
        for (int j = 2; j < cols - 2; ++j) {
            int sum = 0;
            for (int k = -2; k <= 2; ++k) {
                for (int l = -2; l <= 2; ++l) {
                    sum += input[i + k][j + l];
                }
            }
            output[i][j] = sum / 25;
        }
    }
}
int main() {
    std::vector<std::vector<int>> input = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25},       
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25},      
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25},
    };

    int rows = input.size();
    int cols = input[0].size();

    std::vector<std::vector<int>> output(rows, std::vector<int>(cols, 0));

    lowPassFilter(input, output);

    // Print the filtered output
    for (const auto& row : output) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}*/
