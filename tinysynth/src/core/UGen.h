// UGen.h
#ifndef UGEN_H
#define UGEN_H

#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

namespace tinysynth {

template <typename sample_type> class UGen {
public:
    UGen() = default;
    UGen(const UGen &) = default;
    UGen(UGen &&) = delete;
    UGen &operator=(const UGen &) = default;
    UGen &operator=(UGen &&) = delete;
    virtual ~UGen() = default;

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
    [[nodiscard]] virtual std::unique_ptr<UGen> clone() const = 0;
    virtual void reset() = 0;
    virtual void prepare(unsigned int sampleRate) {}

protected:
    template <typename T> static T clamp(T value, T min, T max) {
        return std::max(min, std::min(value, max));
    }
};

} // namespace tinysynth

#endif // UGEN_H