#pragma once

#include "UGen.h"
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <immintrin.h> // AVX

namespace tinysynth {

template <typename sample_type>
struct SIMDOperations {
    static constexpr int vector_size = sizeof(__m256) / sizeof(sample_type);

    static void multiply(const sample_type* input, sample_type* output, unsigned int numFrames, sample_type gain) {
        unsigned int i = 0;

        __m256 gain_avx = _mm256_set1_ps(gain);

        for (; i + vector_size <= numFrames; i += vector_size) {
            __m256 in = _mm256_loadu_ps(&input[i]);
            __m256 out = _mm256_mul_ps(in, gain_avx);
            _mm256_storeu_ps(&output[i], out);
        }

        for (; i < numFrames; ++i) {
            output[i] = input[i] * gain;
        }
    }
};

template <typename sample_type>
class GainModule : public tinysynth::UGen<sample_type> {
public:
    GainModule() : m_gain(1.0F) {}

    static constexpr float MAX_GAIN = 10.0F;

    void process(const std::vector<sample_type*>& inputs, std::vector<sample_type*>& outputs,
                 unsigned int numFrames) override {
        if (inputs.empty() || outputs.empty() || numFrames == 0) {
            return;
        }

        const sample_type* __restrict input = inputs[0];
        sample_type* __restrict output = outputs[0];

        SIMDOperations<sample_type>::multiply(input, output, numFrames, m_gain);
    }

    [[nodiscard]] unsigned int getNumInputs() const noexcept override { return 1; }

    [[nodiscard]] unsigned int getNumOutputs() const noexcept override { return 1; }

    [[nodiscard]] std::string getInputName(unsigned int index) const override {
        if (index != 0) {
            throw std::out_of_range("Invalid input index");
        }
        return "Input";
    }

    [[nodiscard]] std::string getOutputName(unsigned int index) const override {
        if (index != 0) {
            throw std::out_of_range("Invalid output index");
        }
        return "Output";
    }

    void setParameter(const std::string& name, sample_type value) override {
        if (name == "Gain") {
            m_gain = std::clamp(value, 0.0F, MAX_GAIN);
        }
    }

    [[nodiscard]] sample_type getParameter(const std::string& name) const override {
        if (name == "Gain") {
            return m_gain;
        }
        return 0.0F;
    }

    [[nodiscard]] std::vector<std::string> getParameterNames() const noexcept override {
        return { "Gain" };
    }

    [[nodiscard]] std::string getName() const noexcept override { return "Gain Module"; }

    [[nodiscard]] std::string getDescription() const noexcept override {
        return "A simple gain control module.";
    }

    [[nodiscard]] std::unique_ptr<UGen<sample_type>> clone() const override {
        return std::make_unique<GainModule<sample_type>>(*this);
    }

    void reset() noexcept override { m_gain = 1.0F; }

private:
    sample_type m_gain;
};

} // namespace tinysynth
