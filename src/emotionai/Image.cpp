#include "Image.h"
#include <common/base64.h>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <emotiefflib/facial_analysis.h>

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
			// Also provide PNG format if needed
			bundle["image/png"] = encoded; // Note: This assumes JPEG data, might need conversion
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

		// Get the original dimensions
		int originalWidth = inputImage.cols;
		int originalHeight = inputImage.rows;

		if (originalWidth <= targetWidth)
		{
			return inputImage.clone();
		}

		// Calculate the scaling factor
		double scaleFactor = static_cast<double>(targetWidth) / originalWidth;

		// Calculate the new height while maintaining the aspect ratio
		int targetHeight = static_cast<int>(originalHeight * scaleFactor);

		// Resize the image
		cv::Mat outputImage;
		cv::resize(inputImage, outputImage, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_LINEAR);

		return outputImage;
	}

	// MTCNNDetector implementation (placeholder)
	MTCNNDetector::MTCNNDetector(const Config &pConfig, const Config &rConfig, const Config &oConfig)
	{
		// Initialize MTCNN detector with the provided configurations
		// This would load the Caffe models in a real implementation
		spdlog::info("Initializing MTCNN detector with models: {}, {}, {}",
					 pConfig.caffeModel, rConfig.caffeModel, oConfig.caffeModel);
	}

	std::vector<Face> MTCNNDetector::detect(const cv::Mat &image, float scale_factor, float threshold)
	{
		// Placeholder implementation - in real code, this would run MTCNN detection
		std::vector<Face> faces;

		// For now, use OpenCV's face detection as fallback
		static cv::CascadeClassifier face_cascade;
		static bool cascade_loaded = false;

		if (!cascade_loaded)
		{
			try
			{
				std::string cascade_path = cv::samples::findFile("haarcascade_frontalface_default.xml");
				if (face_cascade.load(cascade_path))
				{
					cascade_loaded = true;
				}
			}
			catch (...)
			{
				spdlog::warn("Could not load Haar cascade for face detection");
			}
		}

		if (cascade_loaded)
		{
			std::vector<cv::Rect> detected_faces;
			cv::Mat gray;
			cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
			cv::equalizeHist(gray, gray);

			face_cascade.detectMultiScale(gray, detected_faces, 1.1, 3, 0, cv::Size(30, 30));

			for (const auto &rect : detected_faces)
			{
				Face face;
				face.bbox.x1 = static_cast<float>(rect.x);
				face.bbox.y1 = static_cast<float>(rect.y);
				face.bbox.x2 = static_cast<float>(rect.x + rect.width);
				face.bbox.y2 = static_cast<float>(rect.y + rect.height);
				faces.push_back(face);
			}
		}

		return faces;
	}

	std::vector<cv::Mat> Image::recognizeFaces(const cv::Mat &frame, int downscaleWidth)
	{
		// Configuration for MTCNN
		std::string dirWithModels = "./models"; // Adjust this path as needed

		MTCNNDetector::Config pConfig;
		pConfig.protoText = dirWithModels + "/det1.prototxt";
		pConfig.caffeModel = dirWithModels + "/det1.caffemodel";
		pConfig.threshold = 0.6f;

		MTCNNDetector::Config rConfig;
		rConfig.protoText = dirWithModels + "/det2.prototxt";
		rConfig.caffeModel = dirWithModels + "/det2.caffemodel";
		rConfig.threshold = 0.7f;

		MTCNNDetector::Config oConfig;
		oConfig.protoText = dirWithModels + "/det3.prototxt";
		oConfig.caffeModel = dirWithModels + "/det3.caffemodel";
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

			// Ma addition
			// if face.bbox.x1 < 0:
			//     face.bbox.x1 = 0
			// if face.bbox.y1 < 0:
			//     face.bbox.y1 = 0

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

	std::pair<cv::Mat, nlohmann::json> Image::process_image(const cv::Mat &image)
	{
		try
		{
			// Convert to RGB for display and processing
			cv::Mat image_rgb;
			cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

			// Detect faces once and reuse the results
			std::vector<cv::Mat> facial_images = recognizeFaces(image_rgb, 500);

			if (facial_images.empty())
			{
				throw std::runtime_error("no_faces_detected");
			}

			std::string backend = "torch"; // ["onnx", "torch"]
			std::string modelName = EmotiEffLib::getSupportedModels(backend)[4];
			std::string modelPath = "/home/alex/git/EmotionAI/contrib/emotiefflib/models/affectnet_emotions/"+modelName;
			auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend, modelPath);

			// Process emotions for all detected faces
			std::vector<EmotiEffLib::EmotiEffLibRes> scores_list;
			scores_list.reserve(facial_images.size());

			for (const auto &face_img : facial_images)
			{
				EmotiEffLib::EmotiEffLibRes scores = fer->predictEmotions(face_img, true);
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
			main_pred["label"] = first_score.labels[main_emotion_idx];
			main_pred["probability"] = first_score.scores[main_emotion_idx];

			// Additional probabilities
			nlohmann::json additional_probs;
			for (size_t i = 0; i < first_score.labels.size(); ++i)
			{
				additional_probs[first_score.labels[i]] = fmt::format("{:.2f}", first_score.scores[i]);
			}
			result["additional_probs"] = std::move(additional_probs);

			cv::Mat annotated_image = image_rgb.clone();
			for (const auto &face_img : facial_images)
			{
				// Simplified: draw rectangle around the entire image area where face was detected
				// This is a placeholder - actual implementation should use original face coordinates
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