#ifndef CAMERAEXCEPTION_H_
#define CAMERAEXCEPTION_H_

class CameraException : public std::exception {
public:
  const char *what() const noexcept override {
    return "Error: No camera detected.";
  }
};

#endif
