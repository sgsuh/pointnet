#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <string>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "mat.h"
#include "net.h"

struct Config {
    std::string data_file;
    std::string model_param;
    std::string model_bin;

    std::string input_layer; 
    std::string output_layer;
};

Config load_config(const std::string& config_file) {
    Config cfg;
    try {
        YAML::Node config = YAML::LoadFile(config_file);

        if (config["paths"]) {
            const YAML::Node& paths = config["paths"];
            
            cfg.data_file = paths["data_file"].as<std::string>();
            cfg.model_param = paths["model_param"].as<std::string>();
            cfg.model_bin = paths["model_bin"].as<std::string>();
            
        } else {
            throw std::runtime_error("Configuration file missing 'paths' section.");
        }

        if (config["model_names"]) {
            const YAML::Node& names = config["model_names"];
            cfg.input_layer = names["input_layer"].as<std::string>();
            cfg.output_layer = names["output_layer"].as<std::string>();
        } else {
            throw std::runtime_error("Configuration file missing 'model_names' section.");
        }
        
    } catch (const YAML::BadFile& e) {
        throw std::runtime_error("Could not open or find config file: " + config_file);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML Parsing Error: " + std::string(e.what()));
    }
    return cfg;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_config.yaml>" << std::endl;
        return 1;
    }

    const std::string config_path = argv[1];

    Config cfg;

    try {
        cfg = load_config(config_path); 
        std::cout << "Configuration loaded successfully from: " << config_path << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Configuration Error: " << e.what() << std::endl;
        return 1;
    }

    const char* datapath = cfg.data_file.c_str();

    FILE* fd = fopen(datapath, "rb");

    if(fd == NULL) {
        std::cerr << "File is not found: " << datapath << std::endl;
        return 1;
    }

    fseek(fd, 0, SEEK_END);

    int nLen = ftell(fd);

    fseek(fd, 0, SEEK_SET);

    std::vector<float> points;
    points.resize(nLen / sizeof(float));
    fread(points.data(), 1, nLen, fd);

    fclose(fd);

    // Point Sampler
    std::vector<std::vector<float>> samplePoints(3, std::vector<float>(128));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, nLen / 16);

    float mean[3] = {0.f, 0.f, 0.f};

    for(int i = 0; i < 128; ++i) {
        int idx = dis(gen) * 4;
        samplePoints[0][i] = points[idx];
        samplePoints[1][i] = points[idx + 1];
        samplePoints[2][i] = points[idx + 2];

        mean[0] += points[idx];
        mean[1] += points[idx + 1];
        mean[2] += points[idx + 2];
    }

    // Normalize
    std::vector<float> norm(128, 0.f);
    float maxNorm = 0.0f;

    mean[0] /= 128;
    mean[1] /= 128;
    mean[2] /= 128;

    for(int i = 0; i < 128; ++i) {
        samplePoints[0][i] -= mean[0];
        samplePoints[1][i] -= mean[1];
        samplePoints[2][i] -= mean[2];

        norm[i] += samplePoints[0][i] * samplePoints[0][i];
        norm[i] += samplePoints[1][i] * samplePoints[1][i];
        norm[i] += samplePoints[2][i] * samplePoints[2][i];

        norm[i] = std::sqrt(norm[i]);

        if(norm[i] > maxNorm) {
            maxNorm = norm[i];
        }
    }

    for(int i = 0; i < 128; ++i) {
        samplePoints[0][i] /= maxNorm;
        samplePoints[1][i] /= maxNorm;
        samplePoints[2][i] /= maxNorm;
    }

    ncnn::Mat in(128, 3);

    for(int h = 0; h < in.h; ++h) {
        for(int w = 0; w < in.w; ++w) {
            int idx = (h * in.w + w) * sizeof(float);

            memcpy(in.data + idx, &samplePoints[h][w], sizeof(float));
        }
    }

    ncnn::Net pointnet;
    ncnn::Mat out;

    pointnet.load_param(cfg.model_param.c_str());
    pointnet.load_model(cfg.model_bin.c_str());

    ncnn::Extractor ex = pointnet.create_extractor();

    ex.input(cfg.input_layer.c_str(), in);
    ex.extract(cfg.output_layer.c_str(), out);

    for(int i = 0; i < out.w; ++i) {
        std::cout << out[i] << std::endl;
    }

    return 0;
}
