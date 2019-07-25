#ifndef DFXML_CREATOR_H
#define DFXML_CREATOR_H

#include <pugixml.hpp>
#include <math.h>
#include <dlib/image_processing.h>

// Function declaration (like before main)
class DFXMLCreator
{
private:
	int openssl_sha1(char *name, unsigned char *out);

public:

	void add_DFXML_creator(pugi::xml_node &parent,
						    const char *program_name,
						    const char *program_version);

	void add_fileobject(pugi::xml_node &parent,
						const char *file_path,
						const int number_faces,
						const long original_width,
						const long original_height,
						const long working_width,
						const long working_height,
						const std::vector<dlib::mmod_rect, std::allocator<dlib::mmod_rect>> detected_faces);

	void add_recognition_match(pugi::xml_node &file_object_node,
						const char * file_path,
                        const char * match_image_name,
                        dlib::mmod_rect face_location,
                        const double distance  
                        );

	void add_detection(pugi::xml_node &file_object_node, dlib::mmod_rect face);

	/*
		Includes "fileobject" node creation
		file information - name, size, hash

		Returns file object
	*/
	pugi::xml_node addFileObject(pugi::xml_node &file_object_node, const char * file_path);
	pugi::xml_document create_document();
};

#endif