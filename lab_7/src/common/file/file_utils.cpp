#include "opencv2/opencv.hpp"
#include "file_utils.h"
#include "../logger/logger.h"

#include <string>
#include <fstream>
#include <sstream>

std::string FileUtils::readFile(const std::string &fileName)
{
  std::ifstream file(fileName);
  if (file.is_open()) {
    std::stringstream fileStream;
    fileStream << file.rdbuf();
    file.close();
    return fileStream.str();
  }
  else {
    ERROR("Failed to open file {}. Returning empty string.", fileName);
    return "";
  }
}

cv::Mat FileUtils::readImage(const std::string &fileName, const cv::ImreadModes mode)
{
  return imread(fileName, mode);
}

#include <chrono>
#include <sys/time.h>
#include <ctime>
#include <sstream>
#include <filesystem>
#include <unistd.h>

using std::chrono::duration_cast;
using std::chrono::seconds;
using std::chrono::system_clock;
using std::filesystem::current_path;

#include "paths.h"

// this function is not exposed. if needed, do so
std::string nextImageName()
{
  // Generate a file name bound to be unique
  const auto secSinceEpoch = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
  std::stringstream fName;
  fName << secSinceEpoch << ".bmp";
  // Make a path to the file
  std::filesystem::path path;
  path += current_path();
  path += "/assets/exports/";
  path += fName.str();

  return path.string();
}

void FileUtils::saveImage(const cv::Mat &img, const std::string &fileName)
{
  imwrite(fileName, img);
  DEBUG("Export {}", fileName);
}

void FileUtils::quickSave(const cv::Mat &img)
{
  std::string fileName = nextImageName();
  FileUtils::saveImage(img, fileName);
}
