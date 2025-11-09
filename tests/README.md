# EmotionAI Test Suite

This directory contains unit and integration tests for the EmotionAI application using Google Test.

## Test Structure

- **unit/**: Unit tests for individual components
  - **config/**: Configuration system tests
  - **logging/**: Logger tests
  - **db/**: Redis manager tests
  - **emotionai/**: Image and file processing tests
  - **server/**: Server factory tests

- **integration/**: Integration tests
  - **server/**: Server integration tests
  - **end_to_end/**: End-to-end workflow tests

- **mocks/**: Mock classes for testing

## Running Tests

### Prerequisites
- Google Test installed
- Redis server (for some integration tests)
- OpenCV with test images

### Build and Run
```bash
mkdir build
cd build
cmake ..
make emotionai_tests
./emotionai_tests