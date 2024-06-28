// Module.h
#ifndef MODULE_H
#define MODULE_H

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

template <typename sample_type> class Module {
public:
  Module(const Module &) = default;

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
  virtual void process(const std::vector<sample_type *> &inputs,
                       std::vector<sample_type *> &outputs,
                       unsigned int numFrames) = 0;
  /**
   * @brief Get the number of inputs.
   *
   * @return Number of inputs.
   */
  [[nodiscard]] virtual unsigned int getNumInputs() const = 0;

  /**
   * @brief Get the name of an input.
   *
   * @param index Index of the input.
   * @return Name of the input.
   */
  [[nodiscard]] virtual unsigned int getNumOutputs() const = 0;

  /**
   * @brief Get the name of an output.
   *
   * @param index Index of the output.
   * @return Name of the output.
   */
  [[nodiscard]] virtual std::string getOutputName(unsigned int index) const = 0;

  [[nodiscard]] virtual std::string getInputName(unsigned int index) const = 0;

  /**
   * @brief Set a parameter by name.
   *
   * @param name Name of the parameter.
   * @param value Value of the parameter.
   */
  virtual void setParameter(const std::string &name, sample_type value) = 0;

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
  virtual void prepare(unsigned int sampleRate) {}

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

#endif // MODULE_H

} // namespace tinysynth
