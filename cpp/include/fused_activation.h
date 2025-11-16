#pragma once

#include <cmath>

#include "mat.h"

static inline float activation_ss(float v, int activation_type, const ncnn::Mat& activation_params)
{
    switch(activation_type) {
        case 1:
        {
            v = std::fmax(v, 0.f);

            break;
        }

        case 2:
        {
            float slope = activation_params[0];
            v = v > 0.f ? v : v * slope;

            break;
        }

        case 3:
        {
            float min = activation_params[0];
            float max = activation_params[1];

            if(v < min) {
                v = min;
            }

            if(v > max) {
                v = max;
            }

            break;
        }

        case 4:
        {
            v = std::min(v, 88.3762626647949f);
            v = std::max(v, -88.3762626647949f);
            v = 1.f / (1.f + std::exp(-v));

            break;
        }

        case 5:
        {
            v = v * std::tanh(std::log(std::exp(v) + 1.f));

            break;
        }

        case 6:
        {
            float alpha = activation_params[0];
            float beta = activation_params[1];
            float lower = -beta / alpha;
            float upper = (1.f / alpha) + lower;

            if(v < lower) {
                v = 0.f;
            }
            else if(v > upper) {
                ;
            }
            else {
                v = v * (v * alpha + beta);
            }

            break;
        }
    }

    return v;
}