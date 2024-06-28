#include <immintrin.h> // For AVX
#include <xmmintrin.h> // For SSE

#pragma once

#include <algorithm> // For std::max and std::min
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinysynth {

class Module {
public:
  /**
   * @brief Copy constructor.
   */
  Module(const Module &) = default;

  /**
   * @brief Move constructor is deleted to prevent moving of Module objects.
   * It's often a good choice for base classes
   */
  Module(Module &&) = delete;

  /**
   * @brief Copy assignment operator.
   */
  Module &operator=(const Module &) = default;

  /**
   * @brief Move assignment operator is deleted to prevent moving of Module
   * objects.
   */
  Module &operator=(Module &&) = delete;

  /**
   * @brief Virtual destructor.
   */
  virtual ~Module() = default;

  /**
   * @brief Process audio.
   *
   * @param inputs Vector of input buffers.
   * @param outputs Vector of output buffers.
   * @param numFrames Number of frames to process.
   */
  virtual void process(const std::vector<float *> &inputs,
                       std::vector<float *> &outputs, int numFrames) = 0;
  /**
   * @brief Get the number of inputs.
   *
   * @return Number of inputs.
   */
  [[nodiscard]] virtual int getNumInputs() const = 0;

  /**
   * @brief Get the name of an input.
   *
   * @param index Index of the input.
   * @return Name of the input.
   */
  [[nodiscard]] virtual int getNumOutputs() const = 0;

  /**
   * @brief Get the name of an output.
   *
   * @param index Index of the output.
   * @return Name of the output.
   */
  [[nodiscard]] virtual std::string getOutputName(int index) const = 0;

  [[nodiscard]] virtual std::string getInputName(int index) const = 0;

  /**
   * @brief Set a parameter by name.
   *
   * @param name Name of the parameter.
   * @param value Value of the parameter.
   */
  virtual void setParameter(const std::string &name, float value) = 0;

  /**
   * @brief Get the value of a parameter by name.
   *
   * @param name Name of the parameter.
   * @return Value of the parameter.
   */
  [[nodiscard]] virtual float getParameter(const std::string &name) const = 0;

  /**
   * @brief Get a list of parameter names.
   *
   * @return Vector of parameter names.
   */
  [[nodiscard]] virtual std::vector<std::string> getParameterNames() const = 0;

  /**
   * @brief Get the name of the module.
   *
   * @return Name of the module.
   */
  [[nodiscard]] virtual std::string getName() const = 0;

  /**
   * @brief Get the description of the module.
   *
   * @return Description of the module.
   */
  [[nodiscard]] virtual std::string getDescription() const = 0;

  /**
   * @brief Clone the module (for polyphony or multiple instances).
   *
   * @return Unique pointer to the cloned module.
   */
  [[nodiscard]] virtual std::unique_ptr<Module> clone() const = 0;

  /**
   * @brief Reset the module's internal state.
   */
  virtual void reset() = 0;

  /**
   * @brief Prepare for processing (e.g., update internal buffers based on
   * sample rate).
   *
   * @param sampleRate Sample rate to prepare for.
   */
  virtual void prepare(int sampleRate) {}

protected:
  /**
   * @brief Utility function for clamping values.
   *
   * @tparam T Type of the value.
   * @param value Value to clamp.
   * @param min Minimum allowed value.
   * @param max Maximum allowed value.
   * @return Clamped value.
   */
  template <typename T> static T clamp(T value, T min, T max) {
    return std::max(min, std::min(value, max));
  }
};

/**
 * @brief Example derived class: GainModule.
 */
class GainModule : public Module {
public:
  /**
   * @brief Constructor.
   */
  GainModule();

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

  void process(const std::vector<float *> &inputs,
               std::vector<float *> &outputs, int numFrames) override {
    if (inputs.empty() || outputs.empty() || numFrames <= 0) {
      return;
    }

    const float *input = inputs[0];
    float *output = outputs[0];

#if defined(__AVX__)
    // AVX optimization
    const __m256 gain_vec = _mm256_set1_ps(m_gain);
    for (int i = 0; i < numFrames - 7; i += 8) {
      __m256 in = _mm256_loadu_ps(&input[i]);
      __m256 out = _mm256_mul_ps(in, gain_vec);
      _mm256_storeu_ps(&output[i], out);
    }
    // Handle remaining samples
    for (int i = numFrames - numFrames % 8; i < numFrames; ++i) {
      output[i] = input[i] * m_gain;
    }
#elif defined(__SSE__)
    // SSE optimization
    const __m128 GAIN_VEC = _mm_set1_ps(m_gain);
    for (int i = 0; i < numFrames - 3; i += 4) {
      __m128 in = _mm_loadu_ps(&input[i]);
      __m128 out = _mm_mul_ps(in, GAIN_VEC); // x86 SSE multiply
      _mm_storeu_ps(&output[i], out);
    }
    // Handle remaining samples
    for (int i = numFrames - numFrames % 4; i < numFrames; ++i) {
      output[i] = input[i] * m_gain;
    }
#else
    // Fallback to scalar processing
    for (int i = 0; i < numFrames; ++i) {
      output[i] = input[i] * m_gain;
    }
#endif
  }

  /**
   * @brief Get the number of inputs.
   * 
   * @return Number of inputs.
   */
  [[nodiscard]] int getNumInputs() const override { return 1; }

  /**
   * @brief Get the number of outputs.
   * 
   * @return Number of outputs.
   */
  [[nodiscard]] int getNumOutputs() const override { return 1; }

  /**
   * @brief Get the name of an input.
   * 
   * @param index Index of the input.
   * @return Name of the input.
   */
  [[nodiscard]] std::string getInputName(int index) const override {
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
  [[nodiscard]] std::string getOutputName(int index) const override {
    return "Output";
  }

  /**
   * @brief Set a parameter by name.
   * 
   * @param name Name of the parameter.
   * @param value Value of the parameter.
   */
  void setParameter(const std::string &name, float value) override {
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
  [[nodiscard]] float getParameter(const std::string &name) const override {
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
  [[nodiscard]] std::vector<std::string> getParameterNames() const override {
    return {"Gain"};
  }


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
  [[nodiscard]] std::string getDescription() const override {
    return "A simple gain control module.";
  }

  /**
   * @brief Clone the module.
   * 
   * @return Unique pointer to the cloned module.
   */
  [[nodiscard]] std::unique_ptr<Module> clone() const override {
    return std::make_unique<GainModule>(*this);
  }

  /**
   * @brief Reset the module's internal state.
   */
  void reset() override { m_gain = 1.0F; }

private:
  float m_gain{};
};

} // namespace tinysynth
