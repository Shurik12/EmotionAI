#include "Image.h"
#include <common/base64.h>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <emotiefflib/facial_analysis.h>
#include <common/Config.h>
#include <mtcnn/detector.h>

namespace fs = std::filesystem;

namespace EmotionAI
{
	// Image class implementation
	Image::Image(const std::string &filename)
	{
		std::ifstream fin(filename, std::ios::binary);
		if (!fin.is_open())
		{
			throw std::runtime_error("Failed to open file: " + filename);
		}
		m_buffer << fin.rdbuf();
		fin.close();
	}

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

	std::string Image::get_buffer() const
	{
		return m_buffer.str();
	}

	nl::json Image::mime_bundle_repr() const
	{
		auto bundle = nl::json::object();
		try
		{
			std::string encoded = base64_encode(m_buffer.str());
			bundle["image/jpeg"] = encoded;
		}
		catch (const std::exception &e)
		{
			spdlog::error("Error in mime_bundle_repr: {}", e.what());
		}
		return bundle;
	}

	cv::Mat Image::to_cv_mat() const
	{
		std::string buffer_str = m_buffer.str();
		if (buffer_str.empty())
		{
			return cv::Mat();
		}

		std::vector<uchar> buffer(buffer_str.begin(), buffer_str.end());
		return cv::imdecode(buffer, cv::IMREAD_COLOR);
	}

	// Utility functions implementation
	cv::Mat resizeWithAspectRatio(const cv::Mat &img, int target_width, int target_height)
	{
		if (img.empty())
		{
			spdlog::error("Error: Image is empty!");
			return cv::Mat();
		}

		int original_width = img.cols;
		int original_height = img.rows;

		if (original_width == 0 || original_height == 0)
		{
			return cv::Mat();
		}

		double aspect_ratio = static_cast<double>(original_width) / original_height;

		int new_width = target_width;
		int new_height = target_height;

		if (target_width > 0 && target_height == 0)
		{
			new_height = static_cast<int>(target_width / aspect_ratio);
		}
		else if (target_height > 0 && target_width == 0)
		{
			new_width = static_cast<int>(target_height * aspect_ratio);
		}
		else if (target_width == 0 && target_height == 0)
		{
			return img.clone(); // No resizing needed
		}

		cv::Mat resized_img;
		cv::resize(img, resized_img, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
		return resized_img;
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
		auto &config = Common::Config::instance();

		// Get MTCNN configuration from config
		std::string models_dir = config.faceDetectionModelsPath();

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

		MTCNNDetector detector(pConfig, rConfig, oConfig);

		auto scaledFrame = downscaleImageToWidth(frame, downscaleWidth);
		if (scaledFrame.empty())
		{
			return {};
		}

		double downcastRatioW = static_cast<double>(frame.cols) / scaledFrame.cols;
		double downcastRatioH = static_cast<double>(frame.rows) / scaledFrame.rows;

		std::vector<Face> faces = detector.detect(scaledFrame, 20.f, 0.709f);
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
			if (roi.x >= 0 && roi.y >= 0 &&
				roi.x + roi.width <= frame.cols &&
				roi.y + roi.height <= frame.rows)
			{
				cv::Mat face_img = frame(roi).clone();
				cvFaces.push_back(face_img);
			}
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
			auto &config = Common::Config::instance();
			std::string emotion_model_path = config.emotionModelPath();
			std::string model = fs::path(emotion_model_path).filename().string();
			std::vector<std::string> emotions;
			if (model == "enet_b2_7.pt")
				emotions = { "anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise" };
			else
				emotions = { "anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise" };

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
			auto max_it = std::max_element(first_score.scores.begin(), first_score.scores.end());
			int main_emotion_idx = std::distance(first_score.scores.begin(), max_it);

			// Build JSON result
			nlohmann::json result;

			// Main prediction
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
			spdlog::error("Error processing image: {}", e.what());
			throw;
		}
	}
} // namespace EmotionAI