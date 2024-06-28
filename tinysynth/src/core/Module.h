// Module.h
#ifndef MODULE_H
#define MODULE_H

#pragma once

#include <immintrin.h> // For AVX
#include <xmmintrin.h> // For SSE

#include <algorithm> // For std::max and std::min
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

namespace tinysynth {

template <typename sample_type> class Module {
  public:
    Module() = default;
    Module(const Module &) = default;
    Module(Module &&) = delete;
    Module &operator=(const Module &) = default;
    Module &operator=(Module &&) = delete;
    virtual ~Module() = default;

    // virtual void process(const std::vector<sample_type *> &inputs,
    //                      std::vector<sample_type *> &outputs, unsigned int numFrames) = 0;

  virtual void process(const std::vector<std::optional<sample_type*>>& inputs,
                         std::vector<sample_type*>& outputs,
                         unsigned int numFrames) = 0;
    
    
    [[nodiscard]] virtual unsigned int getNumInputs() const = 0;
    [[nodiscard]] virtual unsigned int getNumOutputs() const = 0;
    [[nodiscard]] virtual std::string getInputName(unsigned int index) const = 0;
    [[nodiscard]] virtual std::string getOutputName(unsigned int index) const = 0;
    virtual void setParameter(const std::string &name, sample_type value) = 0;
    [[nodiscard]] virtual float getParameter(const std::string &name) const = 0;
    [[nodiscard]] virtual std::vector<std::string> getParameterNames() const = 0;
    [[nodiscard]] virtual std::string getName() const = 0;
    [[nodiscard]] virtual std::string getDescription() const = 0;
    [[nodiscard]] virtual std::unique_ptr<Module> clone() const = 0;
    virtual void reset() = 0;
    virtual void prepare(unsigned int sampleRate) {}

  protected:
    template <typename T> static T clamp(T value, T min, T max) {
        return std::max(min, std::min(value, max));
    }
};

#endif // MODULE_H

} // namespace tinysynth
