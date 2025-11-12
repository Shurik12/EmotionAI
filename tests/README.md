# Test Configurations

This directory contains configuration files for different testing scenarios:

## Configuration Files

### unit_config.yaml
- **Purpose**: Unit tests with minimal resource usage
- **Port**: 8081
- **Features**: 
  - Smaller file size limits (10MB)
  - Shorter expiration times
  - File logging disabled
  - Redis DB 1

### integration_config.yaml  
- **Purpose**: Integration tests with realistic settings
- **Port**: 8082
- **Features**:
  - Standard file size limits (50MB)
  - Medium expiration times
  - Both console and file logging
  - Redis DB 2

### e2e_config.yaml
- **Purpose**: End-to-end tests with production-like settings
- **Port**: 8083
- **Features**:
  - Larger file size limits (100MB)
  - Production expiration times
  - Comprehensive logging
  - Redis DB 3

## Usage in Tests

```cpp
// In test setup
auto& config = Common::Config::instance();
config.loadFromFile("tests/unit_config.yaml");