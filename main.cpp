// Native
#include <iostream>
#include <windows.h>
#include <streambuf>
#include <typeinfo>
#include <time.h>
#include <ctime>
#include <Lmcons.h>
#include <chrono>

// OpenSSL
#include <openssl/sha.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// DLIB
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/logger.h>
#include <dlib/opencv.h>

// Boost
#include <boost/filesystem.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/version.hpp>
#include <boost/format.hpp>

// PUGI XML Parsing
#include <pugixml.hpp>

/**
 * Sparing some text
*/
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

using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

struct ImageSplit
{
	int X_offset;
	int Y_offset;
	cv::Mat image;

	ImageSplit(int x_off, int y_off, cv::Mat img)
	{
		X_offset = x_off;
		Y_offset = y_off;
		image = img;
	}
};

/**
 *  Function prototypes
*/
std::vector<filesystem::path> fileSearch(string path);

int findFaces(const std::vector<filesystem::path> path_to_images,
			  net_type detector,
			  const shape_predictor sp,
			  std::vector<matrix<rgb_pixel>> &faces,
			  std::list<std::pair<filesystem::path, int>> &mapping,
			  ofstream &output_file,
			  const long min_size,
			  const long max_size,
			  const bool mark,
			  pugi::xml_document &doc,
			  const std::string workspace);

void add_DFXML_creator(pugi::xml_node &parent,
					   const char *program_name,
					   const char *program_version);

std::string xmlescape(const string &xml);

void add_fileobject(pugi::xml_node &parent,
					const char *file_path,
					const int number_faces,
					const long original_width,
					const long original_height,
					const long working_width,
					const long working_height,
					const std::vector<dlib::mmod_rect, std::allocator<dlib::mmod_rect>> detected_faces);

pugi::xml_document create_document();
void arg_parser(const int argc, const char *const argv[], po::variables_map args, po::options_description desc);
int openssl_sha1(char *name, unsigned char *out);
void get_splits(cv::Mat original, std::vector<ImageSplit> &splits, const int maxH, const int maxW);

/**
 *  DEBUGG toggling
*/
#define DEBUG 1

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
class Log_handler
{
  public:
	Log_handler(std::string filename)
	{
		out_file.open(filename);
	}

	void log(
		const string &logger_name,
		const log_level &ll,
		const dlib::uint64 thread_id,
		const char *message_to_log)
	{

		chrono::system_clock::time_point p = chrono::system_clock::now();
		time_t t = chrono::system_clock::to_time_t(p);

		char buffer[20];
		strftime(buffer, 20, "%Y-%m-%d %H:%M:%S", localtime(&t));

		// We print allways for log and stdout
		out_file << buffer << " " << ll << " [" << thread_id << "] " << logger_name << ": " << message_to_log << endl;

		// But only log messages that are of LINFO priority or higher to the console.
		if (ll < LERROR)
		{
			cout << ctime(&t) << "\n"
				 << ll << " [" << thread_id << "] " << logger_name << ": " << message_to_log << endl;
		}
		else
		{
			cerr << ctime(&t) << "\n"
				 << ll << " [" << thread_id << "] " << logger_name << ": " << message_to_log << endl;
		}
	}

	~Log_handler() { out_file.close(); }

  private:
	ofstream out_file;
};

int const STEP_SIZE = 100;

int main(int argc, char **argv)
{
	auto start = std::chrono::high_resolution_clock::now();
	po::variables_map args;
	long min_size, max_size;
	string params_path;
	dlog.set_level(LALL);
	int device = 0;

	po::options_description desc("Allowed options");
	desc.add_options()("help", "")("params", po::value(&params_path), "Path to configuration file")("min", po::value(&min_size)->default_value(1200 * 1200), "Minimum image size (long - default: 1200x1200)")("max", po::value(&max_size)->default_value(2500 * 2500), "Maximum image size (long - default: 2500x2500)")("debug", "Logger will also print all debug mensages")("version", "Program compiling date")("list_devices", "Lists current CUDA enabled devices")("set_device", po::value(&device)->default_value(0), "Use especific CUDA device (int - default: 0)");

	arg_parser(argc, argv, &args, desc);

	if (max_size < 0 || min_size < 0 || max_size < min_size)
	{
		cout << "Invalid image size" << endl;
		exit(1);
	}

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
			exit(2);
		}

		string workspace, imagesToFindPath, positiveImgPath,
			detectorPath, recognitionPath, shapePredictorPath;

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
			doRecognition = json_tree.get<bool>("doRecognition");
		}
		catch (std::exception &e)
		{
			cout << "Error Initiating variables" << endl;
			exit(3);
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
			exit(11);
		}
		dlib::cuda::set_device(device);

		pugi::xml_document dfxml_doc = create_document();
		pugi::xml_node dfxml_node = dfxml_doc.child("dfxml");

		char filename[MAX_PATH];
		DWORD size = GetModuleFileNameA(NULL, filename, MAX_PATH);
		if (size)
		{
			add_DFXML_creator(dfxml_node, filename, "1.0");
		}

		std::vector<filesystem::path> positivePath, imagesPath;
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
		catch (std::exception &e)
		{
			dlog << LERROR << "Erro ao carregar imagens";
			exit(4);
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
				exit(5);
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
			exit(6);
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
			dlog << LERROR << "Erro ao carregar detector";
			exit(7);
		}

		std::vector<matrix<rgb_pixel>> faces;
		std::list<std::pair<filesystem::path, int>> imgToFaces;
		ofstream output_file;
#if DEBUG
		dlog << LDEBUG << "Creating file to store output";
#endif

		output_file.open(workspace + "\\FDRI_faces_found.txt");
		filesystem::path dir(workspace + "\\annotated");
		filesystem::create_directory(dir);

		int num_positive_faces = 0;
		if (doRecognition)
		{
#if DEBUG
			dlog << LDEBUG << "Searching faces in positive images";
#endif
			num_positive_faces = findFaces(positivePath, detector_dnn, sp, faces, imgToFaces, output_file, min_size, max_size, false, pugi::xml_document(), workspace);
			if (!num_positive_faces)
			{
				cout << "ERROR => Didn't find any positive images in provided folder";
				return 8;
			}
		}

#if DEBUG
		dlog << LDEBUG << "Searching faces in target images";
#endif
		int num_faces_found = findFaces(imagesPath, detector_dnn, sp, faces, imgToFaces, output_file, min_size, max_size, true, dfxml_doc, workspace);
		if (num_faces_found == 0)
		{
			cout << "ERROR => No faces found in provided images";
			return 9;
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

			int num_positive_img = positivePath.size();
			std::list<std::pair<filesystem::path, int>>::iterator it = imgToFaces.begin();
			std::advance(it, num_positive_img);

#if DEBUG
			dlog << LDEBUG << "Opening file to store images with target people";
#endif
			int counter = 0;
			output_file.open(workspace + "\\FDRI_wanted.txt");
			int num_matches = 0;
			ofstream matches_file;
			matches_file.open(workspace + "\\FDRI_img_matches.txt");
			for (it; it != imgToFaces.end(); it++)
			{
				int num_faces_in_image = ((std::pair<filesystem::path, int>)*it).second;
				int delta = num_positive_faces + counter;
				for (int i = 0; i < num_faces_in_image; i++)
				{
					for (int j = 0; j < num_positive_faces; j++)
					{
						auto eu_distance = length(face_descriptors[i + delta] - face_descriptors[j]);
						if (eu_distance < 0.6)
						{
							num_matches++;

							std::list<std::pair<filesystem::path, int>>::iterator it_positives = imgToFaces.begin();
							int count_positives = 0;
							for (it_positives; it_positives != imgToFaces.end(); it_positives++)
							{
								int pos_positive_img = count_positives + ((std::pair<filesystem::path, int>)*it_positives).second;
								if (j < pos_positive_img)
								{
									matches_file << "Match: "
												 << ((std::pair<filesystem::path, int>)*it_positives).first.filename().string()
												 << " with " << ((std::pair<filesystem::path, int>)*it).first.filename().string()
												 << ", distance: " << eu_distance << endl;

									dlog << LINFO << "Match: "
										 << ((std::pair<filesystem::path, int>)*it_positives).first.filename().string()
										 << " with " << ((std::pair<filesystem::path, int>)*it).first.filename().string()
										 << ", distance: " << eu_distance;
									break;
								}
								count_positives += ((std::pair<filesystem::path, int>)*it_positives).second;
							}

							output_file << ((std::pair<filesystem::path, int>)*it).first.filename().string() << endl;
						}
					}
				}
				counter += num_faces_in_image;
			}
			clock_t end = clock();
			double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
			dlog << LINFO << "Recognition time " << format("elapsed_secs=%.8f secs") % elapsed_secs;

			dlog << LINFO << "Found " << num_matches << " images with the wanted people";
			output_file.close();
			matches_file.close();
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
		cout << "Erro: " << e.what() << endl;
		return 11;
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	dlog << LINFO << "Execution ended, total time -> " << elapsed.count();

	return 0;
}

int findFaces(const std::vector<filesystem::path> path_to_images,
			  net_type detector,
			  const shape_predictor sp,
			  std::vector<matrix<rgb_pixel>> &faces,
			  std::list<std::pair<filesystem::path, int>> &mapping,
			  ofstream &output_file,
			  const long min_size,
			  const long max_size,
			  const bool mark,
			  pugi::xml_document &doc,
			  const std::string workspace)
{
	int num_faces = 0;
	pugi::xml_node dfxml_node = doc.child("dfxml");
	double upscale_elapsed_time = 0, downsample_elasped_time = 0;

	for (size_t i = 0; i < path_to_images.size(); i++)
	{
		cv::Mat img = cv::imread(path_to_images[i].generic_string());

		std::vector<ImageSplit> img_splits;
		get_splits(img, img_splits, 2000, 2000);

		long working_width = 0, working_height = 0; //img.nc(),  img.nr();
		int num_faces_in_image = 0;
		//vector1.insert(vector1.end(), vector2.begin(), vector2.end());
		dlib::cv_image<dlib::bgr_pixel> temp_img(img);
		dlib::matrix<dlib::rgb_pixel> dlib_image;
		dlib::assign_image(dlib_image, temp_img);
		
		//TODO: libertar o temp_img?
		temp_img.release();
		img.release();


		for (ImageSplit split : img_splits)
		{

			std::vector<dlib::mmod_rect> detected_faces;
			
			dlib::cv_image<dlib::bgr_pixel> cv_img(split.image);
			dlib::matrix<dlib::rgb_pixel> img_split_dlib;
			dlib::assign_image(img_split_dlib, cv_img);
			cv_img.release();

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
			add_fileobject(dfxml_node, path_to_images[i].generic_string().c_str(),
					   num_faces_in_image, 0,
					   0, working_width,
					   working_height, detected_faces);
		}
		if (num_faces_in_image != 0)
		{
			output_file << path_to_images[i].filename().string() + "\n";
			dlib::save_png(dlib_image, workspace + "\\annotated\\" + path_to_images[i].filename().string());
		}
		
		
				 
		mapping.push_back(std::make_pair(path_to_images[i], num_faces_in_image));
		
	}

	return num_faces;
}

std::vector<filesystem::path> fileSearch(string path)
{
	std::vector<filesystem::path> images_path;
	std::vector<string> targetExtensions;

	targetExtensions.push_back(".JPG");
	targetExtensions.push_back(".BMP");
	targetExtensions.push_back(".GIF");
	targetExtensions.push_back(".PNG");

	if (!filesystem::exists(path))
	{
		exit(4);
	}

	for (filesystem::recursive_directory_iterator end, dir(path); dir != end; ++dir)
	{
		string extension = filesystem::path(*dir).extension().generic_string();
		transform(extension.begin(), extension.end(), extension.begin(), ::toupper);
		if (std::find(targetExtensions.begin(), targetExtensions.end(), extension) != targetExtensions.end())
		{
			images_path.push_back(filesystem::path(*dir));
		}
	}
	return images_path;
}

pugi::xml_document create_document()
{
	pugi::xml_document doc;
	doc.load_string("<?xml version='1.0' encoding='UTF-8'?>\n");
	pugi::xml_node dfxml_node = doc.append_child("dfxml");
	dfxml_node.append_attribute("xmlns") = "http://www.forensicswiki.org/wiki/Category:Digital_Forensics_XML";
	dfxml_node.append_attribute("xmlns:dc") = "http://purl.org/dc/elements/1.1/";
	dfxml_node.append_attribute("xmlns:xsi") = "http://www.w3.org/2001/XMLSchema-instance";
	dfxml_node.append_attribute("version") = "1.1.1";

	return doc;
}

void add_DFXML_creator(pugi::xml_node &parent,
					   const char *program_name,
					   const char *program_version)
{
	pugi::xml_node creator_node = parent.append_child("creator");
	creator_node.append_attribute("version") = "1.0";
	creator_node.append_child("program").text().set(program_name);
	creator_node.append_child("version").text().set(program_version);
	pugi::xml_node build_node = creator_node.append_child("build_environment");

#ifdef BOOST_VERSION
	{
		char buf[64];
		snprintf(buf, sizeof(buf), "%d", BOOST_VERSION);
		pugi::xml_node lib_node = build_node.append_child("library");
		lib_node.append_attribute("name") = "boost";
		lib_node.append_attribute("version") = buf;
	}
#endif
	pugi::xml_node lib_node = build_node.append_child("library");
	lib_node.append_attribute("name") = "pugixml";
	lib_node.append_attribute("version") = "1.9";

	lib_node = build_node.append_child("library");
	lib_node.append_attribute("name") = "dlib";
	lib_node.append_attribute("version") = "19.10";

	pugi::xml_node exe_node = creator_node.append_child("execution_environment");

	chrono::system_clock::time_point p = chrono::system_clock::now();
	time_t t = chrono::system_clock::to_time_t(p);
	exe_node.append_child("start_date").text().set(ctime(&t));

	char username[UNLEN + 1];
	DWORD username_len = UNLEN + 1;
	GetUserName(username, &username_len);
	exe_node.append_child("username").text().set(username);
}

void add_fileobject(pugi::xml_node &parent,
					const char *file_path,
					const int number_faces,
					const long original_width,
					const long original_height,
					const long working_width,
					const long working_height,
					const std::vector<dlib::mmod_rect, std::allocator<dlib::mmod_rect>> detected_faces)
{
	boost::filesystem::path p(file_path);
	uintmax_t f_size = boost::filesystem::file_size(p);
	pugi::xml_node file_obj = parent.append_child("fileobject");
	file_obj.append_child("filesize").text().set(f_size);

	string delimiter = "__id__";
	string aux_name(p.filename().string());
	string img_n = aux_name.substr(0, aux_name.find(delimiter));
	file_obj.append_child("filename").text().set(img_n.c_str());
	// incluir as hashs
	unsigned char hash_buff[SHA_DIGEST_LENGTH];
	if (openssl_sha1((char *)file_path, hash_buff))
	{
		dlog << LWARN << "Error getting file hash";
	}
	else
	{
		pugi::xml_node hash_nodeMD5 = file_obj.append_child("hashdigest");
		hash_nodeMD5.append_attribute("type") = "sha1";
		char tmphash[SHA_DIGEST_LENGTH];

		for (size_t i = 0; i < SHA_DIGEST_LENGTH; i++)
		{
			sprintf((char *)&(tmphash[i * 2]), "%02x", hash_buff[i]);
		}

		hash_nodeMD5.text().set(tmphash);
	}

	pugi::xml_node detection_node = file_obj.append_child("facialdetection");
	detection_node.append_child("number_faces").text().set(number_faces);
	std::stringstream ss;
	ss << original_width << "x" << original_height;
	detection_node.append_child("original_size").text().set(ss.str().c_str());
	ss.clear();
	ss.str("");
	ss << working_width << "x" << working_height;
	detection_node.append_child("working_size").text().set(ss.str().c_str());

	for (int i = 1; i <= detected_faces.size(); i++)
	{
		std::stringstream ss;
		rectangle rec = detected_faces[i - 1];
		ss << rec.left() << " "
		   << rec.top() << " "
		   << rec.right() << " "
		   << rec.bottom();

		pugi::xml_node face_node = detection_node.append_child("face");
		face_node.text().set(ss.str().c_str());

		pugi::xml_node score = face_node.append_child("confidence_score");
		// TODO:: Converter / arredondar
		score.text().set(detected_faces[i].detection_confidence);
	}
}

int openssl_sha1(char *name, unsigned char *out)
{
	FILE *f;
	unsigned char buf[8192];
	SHA_CTX sc;
	int err;

	f = fopen(name, "rb");
	if (f == NULL)
	{
		cout << "Couldn't open file" << endl;
		return -1;
	}
	SHA1_Init(&sc);
	for (;;)
	{
		size_t len;

		len = fread(buf, 1, sizeof buf, f);
		if (len == 0)
			break;
		SHA1_Update(&sc, buf, len);
	}
	err = ferror(f);
	fclose(f);
	if (err)
	{
		cout << "Error hashing file" << endl;
		return -1;
	}
	SHA1_Final(out, &sc);
	return 0;
}

void arg_parser(const int argc, const char *const argv[], po::variables_map args, po::options_description desc)
{

	try
	{
		po::store(
			po::parse_command_line(argc, argv, desc),
			args);
	}
	catch (po::error const &e)
	{
		std::cerr << e.what() << '\n';
		exit(EXIT_FAILURE);
	}

	po::notify(args);

	if (args.count("version"))
	{
		std::cout << "Compilation date: " __DATE__ << " " << __TIME__ << std::endl;
		exit(0);
	}
	if (args.count("list_devices"))
	{
		int num_devices = cuda::get_num_devices();
		cout << "Found " << num_devices << " CUDA supported devices" << endl;
		for (int i = 0; i < num_devices; i++)
		{
			cout << "[" << i << "] => " << cuda::get_device_name(i) << endl;
		}
		exit(0);
	}

	if (args.count("help") || argc < 2 || !args.count("params"))
	{
		std::cout << desc << std::endl;
		exit(2);
	}

	if (args.count("set_device"))
	{
		int selected_device = args["set_device"].as<int>();
		cout << "Device [" << selected_device << "] " << cuda::get_device_name(selected_device) << " selected" << endl;
	}

	/*
	if (!args.count("debug"))
	{
#undef DEBUG
#define DEBUG 0
	}
	*/
}

void get_splits(cv::Mat original, std::vector<ImageSplit> &splits, const int maxH, const int maxW)
{
	int height = original.rows, width = original.cols;
	int num_divisionsH = 0, num_divisionsW = 0;

	while (height > maxH)
	{
		num_divisionsH++;
		height /= 2;
	}
	while (width > maxW)
	{
		num_divisionsW++;
		width /= 2;
	}
	height = height % 2 == 0 ? height : height - (height % 2);
	width = width % 2 == 0 ? width : width - (width % 2);

	//cout << "NumDW - " << num_divisionsW * 2 << " NumDH - " << num_divisionsH * 2 << endl;
	if (num_divisionsH == 0 && num_divisionsW == 0) {
		splits.push_back(ImageSplit(0, 0, original));
	} else if (num_divisionsH && num_divisionsW)
	{
		int y = 0;
		for (int i = 0; i < num_divisionsH * 2; i++)
		{
			int x = 0;
			for (int j = 0; j < num_divisionsW * 2; j++)
			{
				splits.push_back(ImageSplit(width * j, height * i, original(cv::Rect(x, y, width, height))));
				x += width;
			}
			y += height;
		}
	}
	else if (num_divisionsH)
	{
		int y = 0;
		for (int i = 0; i < num_divisionsH * 2; i++)
		{
			splits.push_back(ImageSplit(0, height * i, original(cv::Rect(0, y, width, height))));
			y += height;
		}
	}
	else if (num_divisionsW)
	{
		int x = 0;
		for (int j = 0; j < num_divisionsW * 2; j++)
		{
			splits.push_back(ImageSplit(width * j, 0, original(cv::Rect(x, 0, width, height))));

			x += width;
		}
	}
}