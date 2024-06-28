// GainModule.h
#ifndef GAINMODULE_H
#define GAINMODULE_H

#include "Module.h"
#include <immintrin.h> // For AVX and SSE
#include <memory>

namespace tinysynth {

/**
 * @brief Example derived class: GainModule.
 */
template <typename sample_type> class GainModule : public tinysynth::Module<sample_type> {
public:
    /**
     * @brief Constructor.
     */
    GainModule(): m_gain(1.0F) {}

    /**
     * @brief Maximum gain value.
     */
    static constexpr float MAX_GAIN = 10.0F;

    /**
     * @brief Process audio with optional SIMD optimizations.
     *
     * @param inputs Vector of input buffers.
     * @param outputs Vector of output buffers.
     * @param numFrames Number of frames to process.
     */
    void process(const std::vector<sample_type*>& inputs, std::vector<sample_type*>& outputs,
                 unsigned int numFrames) override {
        if (inputs.empty() || outputs.empty() || numFrames == 0) {
            return;
        }

        const sample_type* input = inputs[0];
        sample_type* output = outputs[0];

#if defined(__AVX__)
        // AVX optimization
        const __m256 gain_vec = _mm256_set1_ps(m_gain);
        unsigned int i = 0;
        for (; i < numFrames - 7; i += 8) {
            __m256 in = _mm256_loadu_ps(&input[i]);
            __m256 out = _mm256_mul_ps(in, gain_vec);
            _mm256_storeu_ps(&output[i], out);
        }
        // Handle remaining samples
        for (; i < numFrames; ++i) {
            output[i] = input[i] * m_gain;
        }
#elif defined(__SSE__)
        // SSE optimization
        const __m128 gain_vec = _mm_set1_ps(m_gain);
        unsigned int i = 0;
        for (; i < numFrames - 3; i += 4) {
            __m128 in = _mm_loadu_ps(&input[i]);
            __m128 out = _mm_mul_ps(in, gain_vec);
            _mm_storeu_ps(&output[i], out);
        }
        // Handle remaining samples
        for (; i < numFrames; ++i) {
            output[i] = input[i] * m_gain;
        }
#else
        // Fallback to scalar processing
        for (unsigned int i = 0; i < numFrames; ++i) {
            output[i] = input[i] * m_gain;
        }
#endif
    }

    /**
     * @brief Get the number of inputs.
     *
     * @return Number of inputs.
     */
    [[nodiscard]] unsigned int getNumInputs() const override { return 1; }

    /**
     * @brief Get the number of outputs.
     *
     * @return Number of outputs.
     */
    [[nodiscard]] unsigned int getNumOutputs() const override { return 1; }

    /**
     * @brief Get the name of an input.
     *
     * @param index Index of the input.
     * @return Name of the input.
     */
    [[nodiscard]] std::string getInputName(unsigned int index) const override {
        if (index != 0) {
            throw std::out_of_range("Invalid input index");
        }
        return "Input";
    }

    /**
     * @brief Get the name of an output.
     *
     * @param index Index of the output.
     * @return Name of the output.
     */
    [[nodiscard]] std::string getOutputName(unsigned int index) const override {
        if (index != 0) {
            throw std::out_of_range("Invalid output index");
        }
        return "Output";
    }

    /**
     * @brief Set a parameter by name.
     *
     * @param name Name of the parameter.
     * @param value Value of the parameter.
     */
    void setParameter(const std::string& name, sample_type value) override {
        if (name == "Gain") {
            m_gain = clamp(value, 0.0F, MAX_GAIN);
        }
    }

    /**
     * @brief Get the value of a parameter by name.
     *
     * @param name Name of the parameter.
     * @return Value of the parameter.
     */
    [[nodiscard]] sample_type getParameter(const std::string& name) const override {
        if (name == "Gain") {
            return m_gain;
        }
        return 0.0F;
    }

    /**
     * @brief Get a list of parameter names.
     *
     * @return Vector of parameter names.
     */
    [[nodiscard]] std::vector<std::string> getParameterNames() const override { return { "Gain" }; }

    /**
     * @brief Get the name of the module.
     *
     * @return Name of the module.
     */
    [[nodiscard]] std::string getName() const override { return "Gain Module"; }

    /**
     * @brief Get the description of the module.
     *
     * @return Description of the module.
     */
    [[nodiscard]] std::string getDescription() const override { return "A simple gain control module."; }

    /**
     * @brief Clone the module.
     *
     * @return Unique pointer to the cloned module.
     */
    [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
        return std::make_unique<GainModule<sample_type>>(*this);
    }

    /**
     * @brief Reset the module's internal state.
     */
    void reset() override { m_gain = 1.0F; }

private:
    sample_type m_gain {};
};

} // namespace tinysynth

#endif // GAINMODULE_H
