#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

constexpr float CONFIDENCE_THRESHOLD = 0; // Confidence threshold
constexpr float NMS_THRESHOLD = 0.4;        // Non-maximum suppression threshold - 0.4
constexpr int NUM_CLASSES = 80;             // Number of classes - 80
constexpr int inpWidth = 608;               // Width of network's input image - 608
constexpr int inpHeight = 608;              // Height of network's input image - 608

// colors for bounding boxes
const Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

int main()
{
    vector<string> class_names;
    {
        ifstream class_file("C:\\TEMP\\classes80.txt");
        if (!class_file)
        {
            cerr << "failed to open classes.txt\n";
            return 0;
        }

        string line;
        while (getline(class_file, line))
            class_names.push_back(line);
    }
    
    //string video = "rtsp://192.168.0.30:554/user=admin_password=tlJwpbo6_channel=0_stream=0.sdp?real_stream";
    string video = "C:\\TEMP\\sample.mp4";

    VideoCapture source;
    source.open(video.c_str());

    auto net = readNetFromDarknet("C:\\TEMP\\yolov4-tiny.cfg", "C:\\TEMP\\yolov4-tiny.weights");

    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA_FP16);
    //net.setPreferableTarget(DNN_TARGET_CUDA);

    //net.setPreferableBackend(DNN_BACKEND_OPENCV);
    //net.setPreferableTarget(DNN_TARGET_CPU);

    auto output_names = net.getUnconnectedOutLayersNames();

    double inference_fps = 0;
    double total_fps = 0;
    cout << "Press ESC to exit" << endl;

    Mat frame, blob;
    vector<Mat> detections;
    while (waitKey(1) < 1)
    {
        
        source >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }

        auto total_start = chrono::steady_clock::now();

        
        blobFromImage(frame, blob, 0.00392, Size(inpWidth, inpHeight), Scalar(), true, false, CV_32F);
        net.setInput(blob);

        auto dnn_start = chrono::steady_clock::now();
        net.forward(detections, output_names);
        auto dnn_end = chrono::steady_clock::now();
        
        vector<int> indices[NUM_CLASSES];
        vector<Rect> boxes[NUM_CLASSES];
        vector<float> scores[NUM_CLASSES];
        
        for (auto& output : detections)
        {
            const auto num_boxes = output.rows;
            for (int i = 0; i < num_boxes; i++)
            {
                
                    auto x = output.at<float>(i, 0) * frame.cols;
                    auto y = output.at<float>(i, 1) * frame.rows;
                    auto width = output.at<float>(i, 2) * frame.cols;
                    auto height = output.at<float>(i, 3) * frame.rows;
                    Rect rect(x - width / 2, y - height / 2, width, height);

                    for (int c = 0; c < NUM_CLASSES; c++)
                    {
                        auto confidence = *output.ptr<float>(i, 5 + c);
                        if (confidence >= CONFIDENCE_THRESHOLD)
                        {
                            boxes[c].push_back(rect);
                            scores[c].push_back(confidence);
                        }
                    }
                
            }
        }
        
        for (int c = 0; c < NUM_CLASSES; c++)
            NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
        
        for (int c = 0; c < NUM_CLASSES; c++)
        {
            for (size_t i = 0; i < indices[c].size(); ++i)
            {
                const auto color = colors[c % NUM_COLORS];

                auto idx = indices[c][i];
                const auto& rect = boxes[c][idx];
                rectangle(frame, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                ostringstream label_ss;
                label_ss << class_names[c] << ": " << fixed << setprecision(2) << scores[c][idx];
                auto label = label_ss.str();

                int baseline;
                auto label_bg_sz = getTextSize(label.c_str(), FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                rectangle(frame, Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), Point(rect.x + label_bg_sz.width, rect.y), color, FILLED);
                putText(frame, label.c_str(), Point(rect.x, rect.y - baseline - 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 0));
            }
        }
        
        auto total_end = chrono::steady_clock::now();

        inference_fps = 1000.0 / chrono::duration_cast<chrono::milliseconds>(dnn_end - dnn_start).count();
        total_fps = 1000.0 / chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count();
        ostringstream stats_ss;
        stats_ss << fixed << setprecision(2);
        stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();

        int baseline;
        auto stats_bg_sz = getTextSize(stats.c_str(), FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        rectangle(frame, Point(0, 0), Point(stats_bg_sz.width, stats_bg_sz.height + 10), Scalar(0, 0, 0), FILLED);
        putText(frame, stats.c_str(), Point(0, stats_bg_sz.height + 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255));

        namedWindow("output");
        imshow("output", frame);
    
    }
    cout << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps << endl;

    return 0;
}