#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    double start_time, end_time;  //variables to calculate time of each method
    //Upload image using CV Mat class
    cv::Mat image = cv::imread("E:\\courses\\High Performance Computing\\Project\\lena.png", cv::IMREAD_COLOR);
    if (image.empty())
    {
        cout << "Could not find or open the image" << endl;
        return -1;
    }
    String windowTitle = "Lena"; //Name of the window

    namedWindow(windowTitle); // Create a window

    imshow(windowTitle, image); 

    int rows = image.rows;
    int cols = image.cols;
    int kernel_size, numThreads;
    cout << "Image rows no: " << rows << "  Image cols no: " << cols;
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

        #pragma omp parallel for num_threads(numThreads)
    //Parallelize the outer loop, allowing multiple threads to process different rows of the image concurrently.
    //Iterate over the pixels (rows and cols) of the image 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols ; j++) {

            cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);

            // Perform low-pass filtering operation on the pixel with the Kernel size the user entered.
            //Iterates over a kernel_sizexkernel_size kernel centered around the current pixel
            int blueSum = 0, greenSum = 0, redSum = 0, count = 0;
            for (int k = -border_size; k <= border_size; k++) {
                for (int l = -border_size; l <= border_size; l++) {

                    int row = i + k;
                    int col = j + l;

                    // Check if the current neighbor pixel is within the image 
                    if (row >= 0 && row < rows && col >= 0 && col < cols) {
                        blueSum += image.at<cv::Vec3b>(row, col)[0];
                        greenSum += image.at<cv::Vec3b>(row, col)[1];
                        redSum += image.at<cv::Vec3b>(row, col)[2];
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
    destroyAllWindows(); //destroy all opened windows

    return 0;
}


/*cv::Vec3f sum(0.0f, 0.0f, 0.0f);

                    int row = i + k;
                    int col = j + l;
                    
    cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);

    // Accumulate the pixel values
    sum += cv::Vec3f(pixel[0], pixel[1], pixel[2]);
                }
            }

            // Compute the average
            cv::Vec3b average(sum[0] / (pow(kernel_size, 2)),
                sum[1] / (pow(kernel_size, 2)),
                sum[2] / (pow(kernel_size, 2)));

            // Set the filtered pixel value
            image.at<cv::Vec3b>(i, j) = average; */
            //  int paddingSize = kernel_size / 2; // Padding size for a 5x5 kernel (2 pixels on each side)

               // cv::copyMakeBorder(image, image, paddingSize, paddingSize, paddingSize, paddingSize, cv::BORDER_CONSTANT, cv::Scalar(0));
