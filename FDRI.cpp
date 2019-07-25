// Native
#ifdef _WIN32
	#include <windows.h>
#endif
#include <iostream>

#include <time.h>
#include <ctime>
#include <chrono>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// DLIB
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/logger.h>
#include <dlib/opencv.h>

// Boost
#include <boost/filesystem.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/program_options.hpp>

#include <pugixml.hpp>

#include <FDRI/DFXML_creator.hpp>
#include <FDRI/image_handling.hpp>
#include <helper_functions.hpp>

#define OPENCV

#include "yolo_v2_class.hpp"


using namespace std;
using namespace dlib;
using namespace boost;
using std::ofstream;
namespace po = boost::program_options;

/**
 *  Recognition network definition
*/
template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
												  alevel0<
													  alevel1<
														  alevel2<
															  alevel3<
																  alevel4<
																	  max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;

/**
 *  Detection network definition
*/
template <long num_filters, typename SUBNET>
using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET>
using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET>
using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET>
using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = 
	loss_mmod<
		con<1, 9, 9, 1, 1, 
		rcon5<
			rcon5<
				rcon5<
					downsampler<
						input_rgb_image_pyramid<
							pyramid_down<6>>>>>>>>;

/**
 *  Function prototypes
*/

int findFaces(const std::vector<boost::filesystem::path> path_to_images,
			net_type detector,
			Detector yolo_detector,
			const shape_predictor sp,
			std::vector<matrix<rgb_pixel>> &faces,
			std::list<std::pair<boost::filesystem::path, int>> &mapping,
			ofstream &output_file,
			const long min_size,
			const long max_size,
			const bool mark,
			pugi::xml_document &doc,
			const std::string workspace
			);

void renderImages(const std::vector<ImageSplit> img_vect);

int computeRecognition(
				std::vector<matrix<float, 0, 1>> face_descriptors,
				int num_positive_img,
				int num_positive_faces,
				std::list<std::pair<boost::filesystem::path, int>> imgToFaces,
				std::string workspace,
				pugi::xml_node dfxml_doc
				);

std::string printObject(bbox_t object);


/**
 *  DEBUGG toggling
*/
#define DEBUG 1
//#define IMG_SPLIT 0
/**
 * Global logger object
 * */
logger dlog("FDRI_log");

/**
 *
 * Logging format:
 * [0][1][2][3][4]
 *
 * [0] - Number of milliseconds since program start
 * [1] - Logging level of the message
 * [2] - Thread that printed the message
 * [3] - Logger's name
 * [4] - Message
 *
 */

int const STEP_SIZE = 100;

const int INVALID_SIZE = 1;
const int INVALID_JSON = 2;
const int ERROR_PARSING_JSON = 3;
const int ERROR_LOADING_IMGS = 4;
const int ERROR_LOADING_RECOGNITION_MODEL = 5;
const int ERROR_LOADING_SHAPE_PREDICTION_MODEL = 6;
const int ERROR_LOADING_DETECTION_MODEL = 7;
const int NO_POSITIVE_IMAGES_FOUND = 8;
const int NO_FACES_IN_POSITIVE_IMAGES = 9;
const int NO_CUDA = 11;

const char* VERSION = "1.0";


int main(int argc, char **argv)
{
	auto start = std::chrono::high_resolution_clock::now();
	po::variables_map args;
	long min_size, max_size;
	string params_path;
	dlog.set_level(LALL);
	int device = 0;
    bool render_images = false;
	po::options_description desc("Allowed options");
	desc.add_options()
	("help", "")
	("params", po::value(&params_path), "Path to configuration file")
	("min", po::value(&min_size)->default_value(1200 * 1200), "Minimum image size (long - default: 1200x1200)")
	("max", po::value(&max_size)->default_value(2500 * 2500), "Maximum image size (long - default: 2500x2500)")
	("debug", "Logger will also print all debug mensages")("version", "Program compiling date")
	("list_devices", "Lists current CUDA enabled devices")
	("set_device", po::value(&device)->default_value(0), "Use especific CUDA device (int - default: 0)");

	arg_parser(argc, argv, &args, desc);

	if (max_size < 0 || min_size < 0 || max_size < min_size)
	{
		cout << "Invalid image size" << endl;
		exit(INVALID_SIZE);
	}

	DFXMLCreator creator;


	try
	{
		property_tree::ptree json_tree;
		try
		{
#if DEBUG
			cout << "Loading json configuration" << endl;
#endif
			read_json(params_path, json_tree);
		}
		catch (std::exception &e)
		{
			cout << "Error parsing json" << endl;
			exit(INVALID_JSON);
		}

		string workspace, imagesToFindPath, positiveImgPath,
			detectorPath, recognitionPath, shapePredictorPath,
			yoloConfPath, yoloWeightsPath;

		bool doRecognition;
		try
		{
#if DEBUG
			cout << "Parsing json variables" << endl;
#endif
			workspace = json_tree.get<string>("workspace");
			imagesToFindPath = json_tree.get<string>("imagesPath");
			positiveImgPath = json_tree.get<string>("wanted_faces");
			detectorPath = json_tree.get_child("paths.").get<string>("0");
			recognitionPath = json_tree.get_child("paths.").get<string>("1");
			shapePredictorPath = json_tree.get_child("paths.").get<string>("2");
			yoloConfPath = json_tree.get_child("paths.").get<string>("3");
			yoloWeightsPath = json_tree.get_child("paths.").get<string>("4");
			doRecognition = json_tree.get<bool>("doRecognition");
		}
		catch (std::exception &e)
		{
			cout << "Error Initiating variables" << endl;
			exit(ERROR_PARSING_JSON);
		}

		Log_handler log_hook(string(workspace + "\\FDRI_log.txt"));
		set_all_logging_output_hooks(log_hook);

		dlog << LINFO << "FDRI Starting";
		dlog << LINFO << "Parameter info:"
			<< "\n- Minimum image size: " << min_size
			<< "\n- Maximum image size: " << max_size;

		int nDevices = dlib::cuda::get_num_devices();
		if (nDevices == 0)
		{
			dlog << LERROR << "Didn't find any usable CUDA devices";
			exit(NO_CUDA);
		}
		dlib::cuda::set_device(device);

		DFXMLCreator dfxml_handler;
		pugi::xml_document dfxml_doc = dfxml_handler.create_document();
		pugi::xml_node dfxml_node = dfxml_doc.child("dfxml");

		char filename[MAX_PATH];
		DWORD size = GetModuleFileNameA(NULL, filename, MAX_PATH);
		if (size)
		{
			dfxml_handler.add_DFXML_creator(dfxml_node, filename, VERSION);
		}

		std::vector<boost::filesystem::path> positivePath, imagesPath;
		try
		{
#if DEBUG
			dlog << LINFO << "Searching files for detection";
#endif
			if (doRecognition)
			{
				positivePath = fileSearch(positiveImgPath);
			}

			imagesPath = fileSearch(imagesToFindPath);
		}
		catch (std::exception $e)
		{
			dlog << LERROR << "Error loading images";
			exit(ERROR_LOADING_IMGS);
		}

		dlog << LINFO << "Found: " << imagesPath.size() << " images to search";

		net_type detector_dnn;
		anet_type recognition_dnn;
		shape_predictor sp;

		dlog << LINFO << "Initiating detectors";

		if (doRecognition)
		{
			try
			{
#if DEBUG
				dlog << LDEBUG << "Initializing recognition network";
#endif
				deserialize(recognitionPath) >> recognition_dnn;
			}
			catch (std::exception &e)
			{
				dlog << LERROR << "Error loading: dlib_face_recognition_resnet_model_v1.dat";
				exit(ERROR_LOADING_RECOGNITION_MODEL);
			}
		}

		try
		{
#if DEBUG
			dlog << LDEBUG << "Initializing shape predictor";
#endif
			deserialize(shapePredictorPath) >> sp;
		}
		catch (std::exception &e)
		{
			dlog << LERROR << "Error loading: shape_predictor_5_face_landmarks.dat";
			exit(ERROR_LOADING_SHAPE_PREDICTION_MODEL);
		}

		try
		{
#if DEBUG
			dlog << LDEBUG << "Initializing face detector";
#endif
			deserialize(detectorPath) >> detector_dnn;
		}
		catch (std::exception &e)
		{
			dlog << LERROR << "Error loading detector";
			exit(ERROR_LOADING_DETECTION_MODEL);
		}

#if DEBUG
		dlog << LDEBUG << "Initializing yolo";
#endif
		Detector yolo_detector(yoloConfPath, yoloWeightsPath, device);

		std::vector<matrix<rgb_pixel>> faces;
		std::list<std::pair<boost::filesystem::path, int>> imgToFaces;
		ofstream output_file;
#if DEBUG
		dlog << LDEBUG << "Creating file to store output";
#endif

		output_file.open(workspace + "\\FDRI_faces_found.txt");
		boost::filesystem::path dir(workspace + "\\annotated");
		boost::filesystem::create_directory(dir);

		int num_positive_faces = 0;
		if (doRecognition)
		{
#if DEBUG
			dlog << LDEBUG << "Searching faces in positive images";
#endif
			num_positive_faces = findFaces(positivePath, detector_dnn, yolo_detector, sp, faces, imgToFaces, output_file, min_size, max_size, false, dfxml_doc, workspace);
			if (!num_positive_faces)
			{
				cout << "ERROR => Didn't find any positive images in provided folder";
				exit(NO_POSITIVE_IMAGES_FOUND);
			}
		}

#if DEBUG
		dlog << LDEBUG << "Searching faces in target images";
#endif
		int num_faces_found = 0;
		try {
			num_faces_found = findFaces(imagesPath, detector_dnn, yolo_detector, sp, faces, imgToFaces, output_file, min_size, max_size, true, dfxml_doc, workspace);
			if (num_faces_found == 0)
			{
				cout << "ERROR => No faces found in provided images";
				exit(NO_FACES_IN_POSITIVE_IMAGES);
			}
			dlog << LINFO << "Number of faces extracted: " << num_faces_found;
			output_file.close();

			if (doRecognition)
			{
	#if DEBUG
				dlog << LDEBUG << "Maching people with positive images";
	#endif
				clock_t begin = clock();
				std::vector<matrix<float, 0, 1>> face_descriptors = recognition_dnn(faces);

				pugi::xml_document recognition_dfxml = dfxml_handler.create_document();

				dfxml_handler.add_DFXML_creator(recognition_dfxml, filename, VERSION);
				pugi::xml_node dfxml_node = recognition_dfxml.child("dfxml");
				int num_matches = computeRecognition(
					face_descriptors,
					positivePath.size(),
					num_positive_faces,
					imgToFaces,
					workspace,
					dfxml_node
				);

				clock_t end = clock();
				double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
				dlog << LINFO << "Recognition time " << format("elapsed_secs=%.8f secs") % elapsed_secs;
				dlog << LINFO << "Found " << num_matches << " images with the wanted people";
			}
		} catch(std::exception &e) {
			std::cout << e.what() << std::endl;
		}

#if DEBUG
		dlog << LDEBUG << "Storing dfxml";
#endif
		ofstream os;
		os.open(workspace + "\\dfxml.xml");
		dfxml_doc.save(os);
#if DEBUG
		dlog << LDEBUG << "Closing files";
#endif
		os.close();
	}
	catch (std::exception &e)
	{
		cout << "Error: " << e.what() << endl;
		return 11;
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	dlog << LINFO << "Execution ended, total time -> " << elapsed.count();

	return 0;
}

void renderImages(const std::vector<ImageSplit> img_vect) {
	for (size_t i = 0; i < img_vect.size(); i++) {
	    dlib::matrix<dlib::rgb_pixel> img;
	    cv2Dlib(img_vect[i].image, &img);
		dlib::image_window wind(img, "Image: " + i);
		int dummy;
		cin >> dummy;
	}
}

int findFaces(const std::vector<boost::filesystem::path> path_to_images,
			net_type detector,
			Detector yolo_detector,
			const shape_predictor sp,
			std::vector<matrix<rgb_pixel>> &faces,
			std::list<std::pair<boost::filesystem::path, int>> &mapping,
			ofstream &output_file,
			const long min_size,
			const long max_size,
			const bool mark,
			pugi::xml_document &doc,
			const std::string workspace)
{
	int num_faces = 0;
	// TODO: DFXML node declaration here
	pugi::xml_node dfxml_node = doc.child("dfxml");
	dlog << LINFO << "Starting face detection";
	DFXMLCreator dfxmlHandler;
#ifdef IMG_SPLIT
	for (size_t i = 0; i < path_to_images.size(); i++) {
		cv::Mat img = cv::imread(path_to_images[i].generic_string());

		std::vector<ImageSplit> img_splits;
		get_splits(img, img_splits, 2000, 2000);

		int num_faces_in_image = 0;
		dlib::cv_image<dlib::bgr_pixel> temp_img(img);
		dlib::matrix<dlib::rgb_pixel> dlib_image;
		//dlib::assign_image(dlib_image, temp_img);

		cv2Dlib(img, &dlib_image);
		renderImages(img_splits);

		for (ImageSplit split : img_splits)
		{
			std::vector<dlib::mmod_rect> detected_faces;
			// TODO: Make this threaded
			dlib::cv_image<dlib::bgr_pixel> cv_img(split.image);
			dlib::matrix<dlib::rgb_pixel> img_split_dlib;
			//dlib::assign_image(img_split_dlib, cv_img);

			cv2Dlib(split.image, &img_split_dlib);
			// TODO: -- Increase image size --
			if (img_split_dlib.nr() < 1500 || img_split_dlib.nr() < 1500)
			{
				pyramid_up(img_split_dlib);
			}

			detected_faces = detector(img_split_dlib);

			num_faces_in_image += detected_faces.size();

			for (dlib::mmod_rect face : detected_faces)
			{
				if (mark)
				{
					rectangle rec = face;
					dlib::draw_rectangle(dlib_image,
										 rectangle(
											 rec.left() + split.X_offset,
											 rec.top() + split.Y_offset,
											 rec.right() + split.X_offset,
											 rec.bottom() + split.Y_offset),
										 dlib::rgb_pixel(255, 0, 0), 3);
				}

				auto shape = sp(img_split_dlib, face);
				dlib::matrix<dlib::rgb_pixel> face_chip;
				extract_image_chip(img_split_dlib, get_face_chip_details(shape, 150, 0.25), face_chip);
				faces.push_back(std::move(face_chip));
				num_faces++;
			}

			if (!(i + 1) % STEP_SIZE)
			{
				dlog << LINFO << "Analysing image " << i + 1 << " out of " << path_to_images.size();
			}
			// TODO: DFXML file object creation here
			/*
			add_fileobject(dfxml_node, path_to_images[i].generic_string().c_str(),
					   num_faces_in_image, 0,
					   0, working_width,
					   working_height, detected_faces);
			*/
		}
		if (num_faces_in_image != 0)
		{
			output_file << path_to_images[i].filename().string() + "\n";
			dlib::save_png(dlib_image, workspace + "\\annotated\\" + path_to_images[i].filename().string());
		}
		mapping.push_back(std::make_pair(path_to_images[i], num_faces_in_image));
	}
#else
	int p = 0;
	for (size_t i = 0; i < path_to_images.size(); i++) {
		cv::Mat img;
		try {
			img = cv::imread(path_to_images[i].generic_string());
		} catch(std::exception &e) {
			cout << "Can't read img: " << path_to_images[i] << endl;
			continue;
		}
		std::vector<bbox_t> detections = yolo_detector.detect(img, 0.6);
		pugi::xml_node objNode = dfxmlHandler.addFileObject(dfxml_node, path_to_images[i].generic_string().c_str());
		pugi::xml_node detection_node = objNode.append_child("facialdetection");
		std::vector<dlib::mmod_rect> image_faces;

		dlib::matrix<dlib::rgb_pixel> capturedDlibFrame;
		dlib::cv_image<dlib::bgr_pixel> tmp(img);
		dlib::assign_image(capturedDlibFrame, tmp);

		for(auto obj : detections) {
			// Index for person
			try {

				//std::cout << "Found something" << std::endl;
				if (obj.obj_id == 0) {
					//std::cout << "Human" << std::endl;
					if ((obj.x + obj.w) > img.cols || (obj.y + obj.h) > img.rows) {
						dlog << LINFO << "object is invalid - ignoring: " << printObject(obj);
						continue;
					}

					//std::cout << printObject(obj) << std::endl;
					cv::Rect rec(obj.x, obj.y, obj.w, obj.h);
					cv::Mat objMat(img(rec));
					
					cv::Mat cropped, abc;
					objMat.copyTo(cropped);
					//objMat.copyTo(abc);
					objMat.release();
										
					if (obj.w < 80) {
						cv::resize(cropped, cropped, cv::Size(100, obj.h));
					}
					//cropped.release();
					
					dlib::matrix<dlib::rgb_pixel> dlibFrame;
					//cv2Dlib(abc, &dlibFrame);
							
					dlib::cv_image<dlib::bgr_pixel> temp_img(cropped);
					dlib::assign_image(dlibFrame, temp_img);
				
					std::vector<dlib::mmod_rect> faces_found = detector(dlibFrame);
					
					int offSetX = 0, offSetY = 0;

					//cout << "Numero de faces: " << faces_found.size() << endl;
					for (dlib::mmod_rect face : faces_found)
					{
						dfxmlHandler.add_detection(detection_node, face);
						if (mark)
						{
							rectangle rec = face;
							rectangle globalRec(
								rec.left() + offSetX, 
								rec.top() + offSetY,
								rec.right() + offSetX,
								rec.bottom() + offSetY
							);

							dlib::draw_rectangle(capturedDlibFrame,
												globalRec,
												dlib::rgb_pixel(255, 0, 0), 3);
						}
						auto shape = sp(dlibFrame, face);
						dlib::matrix<dlib::rgb_pixel> face_chip;
						extract_image_chip(dlibFrame, get_face_chip_details(shape, 150, 0.25), face_chip);
						faces.push_back(std::move(face_chip));
						num_faces++;
					}
					// Join the 2 vectors so that later we can report all the faces from the image

					image_faces.insert( image_faces.end(), faces_found.begin(), faces_found.end() );
					mapping.push_back(std::make_pair(path_to_images[i], faces_found.size()));
				}

				
			} catch( std::exception &e) {
				cout << "Erro => " << e.what() << endl;
				dlog << LERROR << "Error obtaining object skiping. Object location: [" << obj.x << "," << obj.y << "][w:" << obj.w << ",h:" << obj.h << "]";
				dlog << LERROR << "Error message: " << e.what();	
			}
		}
		if (image_faces.size() != 0)
		{
			output_file << path_to_images[i].filename().string() + "\n";
			dlib::save_png(capturedDlibFrame, workspace + "\\annotated\\" + path_to_images[i].filename().string());
		}
		img.release();
		p++;
		std::cout << "IMG #" << p << std::endl;
	}

	std::cout << "Total => " << p << std::endl;
#endif
	return num_faces;
}

std::string printObject(bbox_t object) {
	return "[x" + to_string(object.x) + ",y" + to_string(object.y) + "][w" + to_string(object.w) + ",h" + to_string(object.h) + "]";
}

int computeRecognition(
	std::vector<matrix<float, 0, 1>> face_descriptors,
	int num_positive_img,
	int num_positive_faces,
	std::list<std::pair<boost::filesystem::path, int>> imgToFaces,
	std::string workspace,
	pugi::xml_node dfxml_node
) {

	ofstream output_file;
	int counter = 0;
	output_file.open(workspace + "\\FDRI_wanted.txt");
	int num_matches = 0;
	ofstream matches_file;
	matches_file.open(workspace + "\\FDRI_img_matches.txt");

	std::list<std::pair<boost::filesystem::path, int>>::iterator it = imgToFaces.begin();
	std::advance(it, num_positive_img);

	for (it; it != imgToFaces.end(); it++)
	{
		
		int num_faces_in_image = ((std::pair<boost::filesystem::path, int>)*it).second;
		int delta = num_positive_faces + counter;

		for (int i = 0; i < num_faces_in_image; i++)
		{
			for (int j = 0; j < num_positive_faces; j++)
			{
				auto eu_distance = length(face_descriptors[i + delta] - face_descriptors[j]);
				if (eu_distance < 0.6)
				{
					num_matches++;

					std::list<std::pair<boost::filesystem::path, int>>::iterator it_positives = imgToFaces.begin();
					int count_positives = 0;
					for (it_positives; it_positives != imgToFaces.end(); it_positives++)
					{
						int pos_positive_img = count_positives + ((std::pair<boost::filesystem::path, int>)*it_positives).second;
						if (j < pos_positive_img)
						{
							matches_file << "Match: "
										<< ((std::pair<boost::filesystem::path, int>)*it_positives).first.filename().string()
										<< " with " << ((std::pair<boost::filesystem::path, int>)*it).first.filename().string()
										<< ", distance: " << eu_distance << endl;

							dlog << LINFO << "Match: "
								<< ((std::pair<boost::filesystem::path, int>)*it_positives).first.filename().string()
								<< " with " << ((std::pair<boost::filesystem::path, int>)*it).first.filename().string()
								<< ", distance: " << eu_distance;
							break;
						}
						count_positives += ((std::pair<boost::filesystem::path, int>)*it_positives).second;
					}

					output_file << ((std::pair<boost::filesystem::path, int>)*it).first.filename().string() << endl;
				}
			}
		}
		counter += num_faces_in_image;
	}

	output_file.close();
	matches_file.close();

	return num_matches;
}