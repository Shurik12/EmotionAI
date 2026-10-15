#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <emotiefflib/facial_analysis.h>
#include <mutex>

#include <common/base64.h>
#include <config/Config.h>
#include <logging/Logger.h>
#include <mtcnn/detector.h>
#include "Image.h"

namespace fs = std::filesystem;
Image::Image(const cv::Mat &image, const std::string &format)
{
	if (image.empty())
	{
		throw std::runtime_error("Cannot create Image from empty cv::Mat");
	}

	// Encode the image into memory
	std::vector<uchar> buffer;
	if (!cv::imencode(format, image, buffer))
	{
		throw std::runtime_error("Failed to encode image");
	}

	m_buffer.write(reinterpret_cast<const char *>(buffer.data()), buffer.size());
}

cv::Mat downscaleImageToWidth(const cv::Mat &inputImage, int targetWidth)
{
	if (inputImage.empty())
	{
		return cv::Mat();
	}

	int originalWidth = inputImage.cols;
	int originalHeight = inputImage.rows;

	if (originalWidth <= targetWidth)
	{
		return inputImage.clone();
	}

	double scaleFactor = static_cast<double>(targetWidth) / originalWidth;
	int targetHeight = static_cast<int>(originalHeight * scaleFactor);

	cv::Mat outputImage;
	cv::resize(inputImage, outputImage, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_LINEAR);

	return outputImage;
}

std::vector<cv::Mat> Image::recognizeFaces(const cv::Mat &frame, int downscaleWidth)
{
	// The three Caffe networks are immutable per process; load them once and
	// share across calls. A mutex serialises concurrent detect() calls since
	// the underlying Caffe/CV-DNN inference is not thread-safe on the same net.
	static std::mutex detector_mutex;
	static std::unique_ptr<MTCNNDetector> cached_detector;
	static std::string cached_models_dir;

	std::string models_dir = Config::instance().model().face_detection_models_path;

	MTCNNDetector *detector;
	{
		std::lock_guard<std::mutex> lock(detector_mutex);
		if (!cached_detector || cached_models_dir != models_dir)
		{
			ProposalNetwork::Config pConfig;
			pConfig.protoText = (fs::path(models_dir) / "det1.prototxt").string();
			pConfig.caffeModel = (fs::path(models_dir) / "det1.caffemodel").string();
			pConfig.threshold = 0.6f;

			RefineNetwork::Config rConfig;
			rConfig.protoText = (fs::path(models_dir) / "det2.prototxt").string();
			rConfig.caffeModel = (fs::path(models_dir) / "det2.caffemodel").string();
			rConfig.threshold = 0.7f;

			OutputNetwork::Config oConfig;
			oConfig.protoText = (fs::path(models_dir) / "det3.prototxt").string();
			oConfig.caffeModel = (fs::path(models_dir) / "det3.caffemodel").string();
			oConfig.threshold = 0.7f;

			cached_detector = std::make_unique<MTCNNDetector>(pConfig, rConfig, oConfig);
			cached_models_dir = models_dir;
			LOG_INFO("MTCNN detector loaded from {}", models_dir);
		}
		detector = cached_detector.get();
	}

	// Hold the lock for the entire detect() call.
	std::lock_guard<std::mutex> lock(detector_mutex);

	auto scaledFrame = downscaleImageToWidth(frame, downscaleWidth);
	if (scaledFrame.empty())
	{
		return {};
	}

	double downcastRatioW = static_cast<double>(frame.cols) / scaledFrame.cols;
	double downcastRatioH = static_cast<double>(frame.rows) / scaledFrame.rows;

	std::vector<Face> faces = detector->detect(scaledFrame, 20.f, 0.709f);
	std::vector<cv::Mat> cvFaces;
	cvFaces.reserve(faces.size());

	for (auto &face : faces)
	{
		face.bbox.x1 *= downcastRatioW;
		face.bbox.x2 *= downcastRatioW;
		face.bbox.y1 *= downcastRatioH;
		face.bbox.y2 *= downcastRatioH;

		cv::Rect roi(static_cast<int>(face.bbox.x1),
					 static_cast<int>(face.bbox.y1),
					 static_cast<int>(face.bbox.x2 - face.bbox.x1),
					 static_cast<int>(face.bbox.y2 - face.bbox.y1));

		// Ensure ROI is within image bounds
		if (roi.x < 0)
			roi.x = 0;
		if (roi.y < 0)
			roi.y = 0;
		if (roi.x + roi.width > frame.cols)
			roi.width = frame.cols - roi.x;
		if (roi.y + roi.height > frame.rows)
			roi.height = frame.rows - roi.y;

		cv::Mat face_img = frame(roi).clone();
		cvFaces.push_back(face_img);
	}

	return cvFaces;
}

std::pair<cv::Mat, nlohmann::json> Image::process_image(const cv::Mat &image, EmotiEffLib::EmotiEffLibRecognizer *fer)
{
	try
	{
		if (!fer)
		{
			throw std::runtime_error("Emotion recognizer not initialized");
		}

		// Get model from configuration
		auto &config = Config::instance();
		std::string emotion_model_path = config.model().emotion_model_path;
		std::string model = fs::path(emotion_model_path).filename().string();
		std::vector<std::string> emotions;
		if (model == "enet_b2_7.pt")
			emotions = {"anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"};
		else
			emotions = {"anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"};

		// Convert to RGB for display and processing
		cv::Mat image_rgb;
		cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

		// Detect faces
		std::vector<cv::Mat> facial_images = recognizeFaces(image_rgb, 500);

		if (facial_images.empty())
		{
			throw std::runtime_error("no_faces_detected");
		}

		// Process emotions for all detected faces
		std::vector<EmotiEffLib::EmotiEffLibRes> scores_list;
		scores_list.reserve(facial_images.size());

		for (const auto &face_img : facial_images)
		{
			EmotiEffLib::EmotiEffLibRes scores = fer->predictEmotions(face_img, false);
			scores_list.push_back(std::move(scores));
		}

		if (scores_list.empty())
			throw std::runtime_error("no_emotions_detected");

		// Get results for the first face (main face)
		const auto &first_score = scores_list[0];

		// Find the dominant emotion
		const auto range_end = first_score.scores.begin() +
							   (model == "enet_b2_7.pt" ? first_score.scores.size() : first_score.scores.size() - 2);
		const int main_emotion_idx = std::distance(
			first_score.scores.begin(),
			std::max_element(first_score.scores.begin(), range_end));

		// Build JSON result
		nlohmann::json result;

		// Main prediction
		LOG_INFO("main_emotion_idx: {}", main_emotion_idx);
		auto &main_pred = result["main_prediction"];
		main_pred["index"] = main_emotion_idx;
		main_pred["label"] = emotions[main_emotion_idx];
		main_pred["probability"] = first_score.scores[main_emotion_idx];

		// Additional probabilities
		nlohmann::json additional_probs;

		for (size_t i = 0; i < emotions.size(); ++i)
			additional_probs[emotions[i]] = fmt::format("{:.2f}", first_score.scores[i]);

		if (emotions.size() == 8)
		{
			additional_probs["valence"] = fmt::format("{:.2f}", first_score.scores[8]);
			additional_probs["arousal"] = fmt::format("{:.2f}", first_score.scores[9]);
		}

		result["additional_probs"] = std::move(additional_probs);

		cv::Mat annotated_image = image_rgb.clone();
		for (const auto &face_img : facial_images)
		{
			cv::rectangle(annotated_image,
						  cv::Rect(0, 0, face_img.cols, face_img.rows),
						  cv::Scalar(0, 255, 0), 2);
		}

		return {std::move(annotated_image), std::move(result)};
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error processing image: {}", e.what());
		throw;
	}
}