// GainModule.h
#ifndef GAINMODULE_H
#define GAINMODULE_H

#include "Module.h"
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace tinysynth {

/**
 * @brief Example derived class: GainModule.
 */
template <typename sample_type>
class GainModule : public tinysynth::Module<sample_type> {
public:
    /**
     * @brief Constructor.
     */
    GainModule() : m_gain(1.0F) {}

    /**
     * @brief Maximum gain value.
     */
    static constexpr float MAX_GAIN = 10.0F;

    /**
     * @brief Process audio with LLVM SIMD optimizations.
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

        const sample_type* __restrict input = inputs[0];
        sample_type* __restrict output = outputs[0];

        // Use LLVM's vector extensions
        typedef sample_type vector_type __attribute__((vector_size(32)));
        constexpr int vector_size = sizeof(vector_type) / sizeof(sample_type);
        
        const vector_type gain_vec = {m_gain};
        unsigned int i = 0;
        
        // Vector processing
        for (; i + vector_size <= numFrames; i += vector_size) {
            vector_type in;
            __builtin_memcpy(&in, &input[i], sizeof(vector_type));
            vector_type out = in * gain_vec;
            __builtin_memcpy(&output[i], &out, sizeof(vector_type));
        }
        
        // Handle remaining samples
        for (; i < numFrames; ++i) {
            output[i] = input[i] * m_gain;
        }
    }

    /**
     * @brief Get the number of inputs.
     *
     * @return Number of inputs.
     */
    [[nodiscard]] unsigned int getNumInputs() const noexcept override { return 1; }

    /**
     * @brief Get the number of outputs.
     *
     * @return Number of outputs.
     */
    [[nodiscard]] unsigned int getNumOutputs() const noexcept override { return 1; }

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
            m_gain = std::clamp(value, 0.0F, MAX_GAIN);
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
    [[nodiscard]] std::vector<std::string> getParameterNames() const noexcept override {
        return { "Gain" };
    }

    /**
     * @brief Get the name of the module.
     *
     * @return Name of the module.
     */
    [[nodiscard]] std::string getName() const noexcept override { return "Gain Module"; }

    /**
     * @brief Get the description of the module.
     *
     * @return Description of the module.
     */
    [[nodiscard]] std::string getDescription() const noexcept override {
        return "A simple gain control module.";
    }

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
    void reset() noexcept override { m_gain = 1.0F; }

private:
    sample_type m_gain;
};

} // namespace tinysynth

#endif // GAINMODULE_H