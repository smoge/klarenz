#pragma once

#include "../core/Module.h"
#include "../utils/Constants.h"
#include "../utils/Utils.h"
#include "AudioEngine.h"
#include "OscillatorParams.h"
#include <cmath>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/MathExtras.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinysynth {

// Forward declarations
template <typename sample_type> class SineOsc;
template <typename sample_type> class SawOsc;
template <typename sample_type> class TriangleOsc;
template <typename sample_type> class PulseOsc;

// Oscillator traits
template <typename OscType> struct OscillatorTraits;

template <typename sample_type> struct OscillatorTraits<SineOsc<sample_type>> {
  static constexpr const char *name = "Sine Oscillator";
  static constexpr const char *description = "A sine wave oscillator";
  static constexpr int numInputs = 2;
};

template <typename sample_type> struct OscillatorTraits<SawOsc<sample_type>> {#pragma once

#include "../core/UGen.h"
#include "../utils/Constants.h"
#include "../utils/Utils.h"
#include "AudioEngine.h"
#include "OscillatorParams.h"
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/MathExtras.h>

// This file contains the implementation of the Oscillator class and its derived
// classes With LLVM SIMD optimizations and anti-aliasing techniques

namespace tinysynth {

template <typename sample_type> class Oscillator : public UGen<sample_type> {
public:
  virtual sample_type getCurrentValue() const = 0;

  void setFrequency(sample_type frequency) {
    assert(frequency > 0);
    m_frequency = frequency;
  }

  void setAmplitude(sample_type amplitude) {
    assert(amplitude >= 0);
    m_amplitude = amplitude;
  }

  void setPhase(sample_type phase) { m_phase = phase; }

  void setParameter(const std::string &name, sample_type value) override {
    if (name == "frequency") {
      setFrequency(value);
    } else if (name == "amplitude") {
      setAmplitude(value);
    } else if (name == "phase") {
      setPhase(value);
    } else {
      throw std::invalid_argument("Unknown parameter: " + name);
    }
  }

  sample_type getParameter(const std::string &name) const override {
    if (name == "frequency") {
      return getFrequency();
    }
    if (name == "amplitude") {
      return getAmplitude();
    }
    if (name == "phase") {
      return getPhase();
    }
    throw std::invalid_argument("Unknown parameter: " + name);
  }

  [[nodiscard]] std::vector<std::string> getParameterNames() const override {
    return {"frequency", "amplitude", "phase"};
  }

  [[nodiscard]] std::string getInputName(unsigned int index) const override {
    switch (index) {
    case 0:
      return "Frequency Modulation";
    case 1:
      return "Amplitude Modulation";
    default:
      return "";
    }
  }

  [[nodiscard]] std::string getOutputName(unsigned int index) const override {
    return (index == 0) ? "output" : "";
  }

  [[nodiscard]] unsigned int getNumInputs() const override {
    return 2; // Frequency and amplitude modulation inputs
  }

  [[nodiscard]] unsigned int getNumOutputs() const override { return 1; }

  void reset() override { m_phase = 0; }

  void prepare(unsigned int sampleRate) override {
    AudioEngine::setSampleRate(sampleRate);
  }

  sample_type getFrequency() const { return m_frequency; }

  sample_type getAmplitude() const { return m_amplitude; }

  sample_type getPhase() const { return m_phase; }

  virtual ~Oscillator() = default;

  Oscillator(const Oscillator &) = delete;
  Oscillator(Oscillator &&) = delete;
  Oscillator &operator=(const Oscillator &) = delete;
  Oscillator &operator=(Oscillator &&) = delete;

protected:
  // PolyBLEP anti-aliasing function
  sample_type polyBlep(sample_type time, sample_type deltaTime) const {
    if (time < deltaTime) {
      time /= deltaTime;
      return time + time - time * time - 1.0;
    }
    if (time > 1.0 - deltaTime) {
      time = (time - 1.0) / deltaTime;
      return time * time + time + time + 1.0;
    }
    return 0.0;
  }

private:
  sample_type m_frequency{440.0};
  sample_type m_amplitude{1.0};
  sample_type m_phase{0.0};
};

///////////////////////// SinOscc class definition /////////////////////////

template <typename sample_type> class SineOsc : public Oscillator<sample_type> {
public:
  void process(const std::vector<std::optional<sample_type *>> &inputs,
               std::vector<sample_type *> &outputs,
               unsigned int numFrames) override {
    if (outputs.empty() || numFrames == 0) {
      return;
    }

    sample_type *output = outputs[0];
    std::optional<sample_type *> freqMod =
        !inputs.empty() ? inputs[0] : std::nullopt;
    std::optional<sample_type *> ampMod =
        inputs.size() > 1 ? inputs[1] : std::nullopt;

    OscillatorParams<sample_type> oscParams{
        this->getPhase(), this->getFrequency(), this->getAmplitude(),
        static_cast<sample_type>(AudioEngine::getSampleRate())};

    SIMDProcessingParams<sample_type> simdParams{output, freqMod, ampMod,
                                                 numFrames};

    processSIMD(simdParams, oscParams);

    this->setPhase(oscParams.phase);
  }

  [[nodiscard]] std::string getName() const override {
    return "Sine Oscillator";
  }
  [[nodiscard]] std::string getDescription() const override {
    return "A sine wave oscillator";
  }
  [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
    return std::make_unique<SineOsc>(*this);
  }
  sample_type getCurrentValue() const override {
    return this->getAmplitude() * std::sin(this->getPhase());
  }

private:
  using vec_t = llvm::SmallVector<sample_type, 8>;

  void processSIMD(SIMDProcessingParams<sample_type> &simdParams,
                   OscillatorParams<sample_type> &oscParams) {
    constexpr int vectorSize = 8;
    const vec_t twoPi(vectorSize, Constants<sample_type>::twoPiConstant);
    const vec_t sampleRateVec(vectorSize, oscParams.sampleRate);

    vec_t phaseVec(vectorSize, oscParams.phase);
    vec_t freqVec(vectorSize, oscParams.baseFrequency);
    vec_t ampVec(vectorSize, oscParams.baseAmplitude);

    for (unsigned int i = 0; i < simdParams.numFrames; i += vectorSize) {
      // Apply modulation
      if (simdParams.freqMod && *simdParams.freqMod) {
        for (int j = 0; j < vectorSize; ++j) {
          freqVec[j] += (*simdParams.freqMod)[i + j];
        }
      }
      if (simdParams.ampMod && *simdParams.ampMod) {
        for (int j = 0; j < vectorSize; ++j) {
          ampVec[j] *= (*simdParams.ampMod)[i + j];
        }
      }

      // Calculate phase increment and update phase
      vec_t phaseIncVec = freqVec * twoPi / sampleRateVec;

      // Calculate sine wave
      vec_t sinVec(vectorSize);
      for (int j = 0; j < vectorSize; ++j) {
        sinVec[j] = std::sin(phaseVec[j]);
      }

      // Apply amplitude
      vec_t outputVec = ampVec * sinVec;

      // Store result
      for (int j = 0; j < vectorSize && i + j < simdParams.numFrames; ++j) {
        simdParams.output[i + j] = outputVec[j];
      }

      // Update phase
      phaseVec += phaseIncVec;
      for (int j = 0; j < vectorSize; ++j) {
        if (phaseVec[j] >= twoPi[j]) {
          phaseVec[j] -= twoPi[j];
        }
      }
    }

    // Store the last phase
    oscParams.phase = phaseVec.back();
  }
};

/////// SawOsc class definition ///////

template <typename sample_type> class SawOsc : public Oscillator<sample_type> {
public:
  void process(const std::vector<std::optional<sample_type *>> &inputs,
               std::vector<sample_type *> &outputs,
               unsigned int numFrames) override {
    if (outputs.empty() || numFrames == 0) {
      return;
    }

    sample_type *output = outputs[0];
    std::optional<sample_type *> freqMod =
        !inputs.empty() ? inputs[0] : std::nullopt;
    std::optional<sample_type *> ampMod =
        inputs.size() > 1 ? inputs[1] : std::nullopt;

    OscillatorParams<sample_type> oscParams{
        this->getPhase(), this->getFrequency(), this->getAmplitude(),
        static_cast<sample_type>(AudioEngine::getSampleRate())};

    SIMDProcessingParams<sample_type> simdParams{output, freqMod, ampMod,
                                                 numFrames};

    processSIMD(simdParams, oscParams);

    this->setPhase(oscParams.phase);
  }

  [[nodiscard]] std::string getName() const override {
    return "Sawtooth Oscillator";
  }
  [[nodiscard]] std::string getDescription() const override {
    return "An anti-aliased sawtooth wave oscillator with LLVM SIMD "
           "optimization";
  }
  [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
    return std::make_unique<SawOsc>(*this);
  }
  sample_type getCurrentValue() const override {
    sample_type t = this->getPhase() / Constants<sample_type>::twoPiConstant;
    sample_type dt = this->getFrequency() /
                     static_cast<sample_type>(AudioEngine::getSampleRate());
    return this->getAmplitude() * (2.0 * t - 1.0 - this->polyBlep(t, dt));
  }

private:
  using vec_t = llvm::SmallVector<sample_type, 8>;

  void processSIMD(SIMDProcessingParams<sample_type> &simdParams,
                   OscillatorParams<sample_type> &oscParams) {
    constexpr int vectorSize = 8;
    const vec_t twoPi(vectorSize, Constants<sample_type>::twoPiConstant);
    const vec_t sampleRateVec(vectorSize, oscParams.sampleRate);
    const vec_t two(vectorSize, 2.0);
    const vec_t one(vectorSize, 1.0);

    vec_t phaseVec(vectorSize, oscParams.phase);
    vec_t freqVec(vectorSize, oscParams.baseFrequency);
    vec_t ampVec(vectorSize, oscParams.baseAmplitude);

    for (unsigned int i = 0; i < simdParams.numFrames; i += vectorSize) {
      // Apply modulation
      if (simdParams.freqMod && *simdParams.freqMod) {
        for (int j = 0; j < vectorSize; ++j) {
          freqVec[j] += (*simdParams.freqMod)[i + j];
        }
      }
      if (simdParams.ampMod && *simdParams.ampMod) {
        for (int j = 0; j < vectorSize; ++j) {
          ampVec[j] *= (*simdParams.ampMod)[i + j];
        }
      }

      // Calculate phase increment and normalize phase
      vec_t phaseIncVec = freqVec * twoPi / sampleRateVec;
      vec_t tVec = phaseVec / twoPi;
      vec_t dtVec = freqVec / sampleRateVec;

      // Calculate sawtooth wave with anti-aliasing
      vec_t sawWave = two * tVec - one;
      sawWave -= polyBlepSIMD(tVec, dtVec);
      sawWave -= polyBlepSIMD(tVec - one, dtVec);

      // Apply amplitude
      vec_t outputVec = ampVec * sawWave;

      // Store result
      for (int j = 0; j < vectorSize && i + j < simdParams.numFrames; ++j) {
        simdParams.output[i + j] = outputVec[j];
      }

      // Update phase
      phaseVec += phaseIncVec;
      for (int j = 0; j < vectorSize; ++j) {
        if (phaseVec[j] >= twoPi[j]) {
          phaseVec[j] -= twoPi[j];
        }
      }
    }

    // Store the last phase
    oscParams.phase = phaseVec.back();
  }

  vec_t polyBlepSIMD(const vec_t &timeVector, const vec_t &deltaTimeVector) {
    vec_t result(timeVector.size(), 0.0);
    for (size_t index = 0; index < timeVector.size(); ++index) {
      if (timeVector[index] < deltaTimeVector[index]) {
        sample_type timeRatio1 = timeVector[index] / deltaTimeVector[index];
        result[index] = timeRatio1 + timeRatio1 - timeRatio1 * timeRatio1 - 1.0;
      } else if (timeVector[index] > 1.0 - deltaTimeVector[index]) {
        sample_type timeRatio2 =
            (timeVector[index] - 1.0) / deltaTimeVector[index];
        result[index] = timeRatio2 * timeRatio2 + timeRatio2 + timeRatio2 + 1.0;
      }
    }
    return result;
  }
};

////// TriangleOsc class definition //////

template <typename sample_type>
class TriangleOsc : public Oscillator<sample_type> {
public:
  void process(const std::vector<std::optional<sample_type *>> &inputs,
               std::vector<sample_type *> &outputs,
               unsigned int numFrames) override {
    if (outputs.empty() || numFrames == 0) {
      return;
    }

    sample_type *output = outputs[0];
    std::optional<sample_type *> freqMod =
        !inputs.empty() ? inputs[0] : std::nullopt;
    std::optional<sample_type *> ampMod =
        inputs.size() > 1 ? inputs[1] : std::nullopt;

    OscillatorParams<sample_type> oscParams{
        this->getPhase(), this->getFrequency(), this->getAmplitude(),
        static_cast<sample_type>(AudioEngine::getSampleRate())};

    SIMDProcessingParams<sample_type> simdParams{output, freqMod, ampMod,
                                                 numFrames};

    processSIMD(simdParams, oscParams);

    this->setPhase(oscParams.phase);
  }

  [[nodiscard]] std::string getName() const override {
    return "Triangle Oscillator";
  }
  [[nodiscard]] std::string getDescription() const override {
    return "An anti-aliased triangle wave oscillator";
  }
  [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
    return std::make_unique<TriangleOsc>(*this);
  }
  sample_type getCurrentValue() const override {
    sample_type t = this->getPhase() / Constants<sample_type>::twoPiConstant;
    sample_type dt = this->getFrequency() /
                     static_cast<sample_type>(AudioEngine::getSampleRate());

    sample_type value = 2.0 * std::abs(2.0 * t - 1.0) - 1.0;
    sample_type blep1 = this->polyBlep(t, dt);
    sample_type blep2 = this->polyBlep(std::fmod(t + 0.5, 1.0), dt);

    return this->getAmplitude() * (value + 4.0 * (blep1 - blep2));
  }

private:
  using vec_t = llvm::SmallVector<sample_type, 8>;

  void processSIMD(SIMDProcessingParams<sample_type> &simdParams,
                   OscillatorParams<sample_type> &oscParams) {
    constexpr int vectorSize = 8;
    const vec_t twoPi(vectorSize, Constants<sample_type>::twoPiConstant);
    const vec_t sampleRateVec(vectorSize, oscParams.sampleRate);
    const vec_t two(vectorSize, 2.0);
    const vec_t one(vectorSize, 1.0);
    const vec_t four(vectorSize, 4.0);
    const vec_t half(vectorSize, 0.5);

    vec_t phaseVec(vectorSize, oscParams.phase);
    vec_t freqVec(vectorSize, oscParams.baseFrequency);
    vec_t ampVec(vectorSize, oscParams.baseAmplitude);

    for (unsigned int i = 0; i < simdParams.numFrames; i += vectorSize) {
      // Apply modulation
      if (simdParams.freqMod && *simdParams.freqMod) {
        for (int j = 0; j < vectorSize; ++j) {
          freqVec[j] += (*simdParams.freqMod)[i + j];
        }
      }
      if (simdParams.ampMod && *simdParams.ampMod) {
        for (int j = 0; j < vectorSize; ++j) {
          ampVec[j] *= (*simdParams.ampMod)[i + j];
        }
      }

      // Calculate phase increment and normalize phase
      vec_t phaseIncVec = freqVec * twoPi / sampleRateVec;
      vec_t tVec = phaseVec / twoPi;
      vec_t dtVec = freqVec / sampleRateVec;

      // Calculate triangle wave with anti-aliasing
      vec_t triWave(vectorSize);
      for (int j = 0; j < vectorSize; ++j) {
        sample_type t = tVec[j];
        sample_type dt = dtVec[j];
        sample_type value = 2.0 * std::abs(2.0 * t - 1.0) - 1.0;
        sample_type blep1 = this->polyBlep(t, dt);
        sample_type blep2 = this->polyBlep(std::fmod(t + 0.5, 1.0), dt);
        triWave[j] = value + 4.0 * (blep1 - blep2);
      }

      // Apply amplitude
      vec_t outputVec = ampVec * triWave;

      // Store result
      for (int j = 0; j < vectorSize && i + j < simdParams.numFrames; ++j) {
        simdParams.output[i + j] = outputVec[j];
      }

      // Update phase
      phaseVec += phaseIncVec;
      for (int j = 0; j < vectorSize; ++j) {
        if (phaseVec[j] >= twoPi[j]) {
          phaseVec[j] -= twoPi[j];
        }
      }
    }

    // Store the last phase
    oscParams.phase = phaseVec.back();
  }
};

//////////////////// Pulse class definition //////////////////////

template <typename sample_type>
class PulseOsc : public Oscillator<sample_type> {
public:
  PulseOsc() : m_pulseWidth(0.5) {}

  void process(const std::vector<std::optional<sample_type *>> &inputs,
               std::vector<sample_type *> &outputs,
               unsigned int numFrames) override {
    if (outputs.empty() || numFrames == 0) {
      return;
    }

    sample_type *output = outputs[0];
    std::optional<sample_type *> freqMod =
        !inputs.empty() ? inputs[0] : std::nullopt;
    std::optional<sample_type *> ampMod =
        inputs.size() > 1 ? inputs[1] : std::nullopt;
    std::optional<sample_type *> pwMod =
        inputs.size() > 2 ? inputs[2] : std::nullopt;

    OscillatorParams<sample_type> oscParams{
        this->getPhase(), this->getFrequency(), this->getAmplitude(),
        static_cast<sample_type>(AudioEngine::getSampleRate())};

    SIMDProcessingParams<sample_type> simdParams{output, freqMod, ampMod, pwMod,
                                                 numFrames};

    processSIMD(simdParams, oscParams);

    this->setPhase(oscParams.phase);
  }

  [[nodiscard]] std::string getName() const override {
    return "Pulse Oscillator";
  }
  [[nodiscard]] std::string getDescription() const override {
    return "An anti-aliased pulse wave oscillator with pulse width control";
  }
  [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
    return std::make_unique<PulseOsc>(*this);
  }

  sample_type getCurrentValue() const override {
    sample_type t = this->getPhase() / Constants<sample_type>::twoPiConstant;
    sample_type dt = this->getFrequency() /
                     static_cast<sample_type>(AudioEngine::getSampleRate());
    sample_type value = (t < m_pulseWidth) ? 1.0 : -1.0;
    value -= this->polyBlep(t, dt);
    value += this->polyBlep(std::fmod(t + 1.0 - m_pulseWidth, 1.0), dt);
    return this->getAmplitude() * value;
  }

  void setPulseWidth(sample_type pulseWidth) {
    m_pulseWidth = std::clamp(pulseWidth, static_cast<sample_type>(0.0),
                              static_cast<sample_type>(1.0));
  }

  sample_type getPulseWidth() const { return m_pulseWidth; }

  void setParameter(const std::string &name, sample_type value) override {
    if (name == "pulseWidth") {
      setPulseWidth(value);
    } else {
      Oscillator<sample_type>::setParameter(name, value);
    }
  }

  sample_type getParameter(const std::string &name) const override {
    if (name == "pulseWidth") {
      return getPulseWidth();
    }
    return Oscillator<sample_type>::getParameter(name);
  }

  [[nodiscard]] std::vector<std::string> getParameterNames() const override {
    auto names = Oscillator<sample_type>::getParameterNames();
    names.push_back("pulseWidth");
    return names;
  }

  [[nodiscard]] unsigned int getNumInputs() const override {
    return 3; // Frequency, amplitude, and pulse width modulation inputs
  }

private:
  using vec_t = llvm::SmallVector<sample_type, 8>;

  void processSIMD(SIMDProcessingParams<sample_type> &simdParams,
                   OscillatorParams<sample_type> &oscParams) {
    constexpr int vectorSize = 8;
    const vec_t twoPiVector(vectorSize, Constants<sample_type>::twoPiConstant);
    const vec_t sampleRateVector(vectorSize, oscParams.sampleRate);
    const vec_t oneVector(vectorSize, 1.0);

    vec_t phaseVector(vectorSize, oscParams.phase);
    vec_t frequencyVector(vectorSize, oscParams.baseFrequency);
    vec_t amplitudeVector(vectorSize, oscParams.baseAmplitude);
    vec_t pulseWidthVector(vectorSize, m_pulseWidth);

    for (unsigned int frameIndex = 0; frameIndex < simdParams.numFrames;
         frameIndex += vectorSize) {
      // Apply modulation
      if (simdParams.freqMod && *simdParams.freqMod) {
        for (int elementIndex = 0; elementIndex < vectorSize; ++elementIndex) {
          frequencyVector[elementIndex] +=
              (*simdParams.freqMod)[frameIndex + elementIndex];
        }
      }
      if (simdParams.ampMod && *simdParams.ampMod) {
        for (int elementIndex = 0; elementIndex < vectorSize; ++elementIndex) {
          amplitudeVector[elementIndex] *=
              (*simdParams.ampMod)[frameIndex + elementIndex];
        }
      }
      if (simdParams.pwMod && *simdParams.pwMod) {
        for (int elementIndex = 0; elementIndex < vectorSize; ++elementIndex) {
          pulseWidthVector[elementIndex] = std::clamp(
              m_pulseWidth + (*simdParams.pwMod)[frameIndex + elementIndex],
              static_cast<sample_type>(0.0), static_cast<sample_type>(1.0));
        }
      }

      // Calculate phase increment and normalize phase
      vec_t phaseIncrementVector =
          frequencyVector * twoPiVector / sampleRateVector;
      vec_t tVector = phaseVector / twoPiVector;
      vec_t deltaTVector = frequencyVector / sampleRateVector;

      // Calculate pulse wave with anti-aliasing
      vec_t pulseWaveVector(vectorSize);
      for (int elementIndex = 0; elementIndex < vectorSize; ++elementIndex) {
        sample_type tValue = tVector[elementIndex];
        sample_type deltaTValue = deltaTVector[elementIndex];
        sample_type pulseWidthValue = pulseWidthVector[elementIndex];

        sample_type waveValue = (tValue < pulseWidthValue) ? 1.0 : -1.0;
        waveValue -= this->polyBlep(tValue, deltaTValue);
        waveValue += this->polyBlep(
            std::fmod(tValue + 1.0 - pulseWidthValue, 1.0), deltaTValue);
        pulseWaveVector[elementIndex] = waveValue;
      }

      // Apply amplitude
      vec_t outputVector = amplitudeVector * pulseWaveVector;

      // Store result
      for (int elementIndex = 0;
           elementIndex < vectorSize &&
           frameIndex + elementIndex < simdParams.numFrames;
           ++elementIndex) {
        simdParams.output[frameIndex + elementIndex] =
            outputVector[elementIndex];
      }

      // Update phase
      phaseVector += phaseIncrementVector;
      for (int elementIndex = 0; elementIndex < vectorSize; ++elementIndex) {
        if (phaseVector[elementIndex] >= twoPiVector[elementIndex]) {
          phaseVector[elementIndex] -= twoPiVector[elementIndex];
        }
      }
    }

    // Store the last phase
    oscParams.phase = phaseVector.back();
  }

  sample_type m_pulseWidth;
};

} // namespace tinysynth

  static constexpr const char *name = "Sawtooth Oscillator";
  static constexpr const char *description =
      "An anti-aliased sawtooth wave oscillator with LLVM SIMD optimization";
  static constexpr int numInputs = 2;
};

template <typename sample_type>
struct OscillatorTraits<TriangleOsc<sample_type>> {
  static constexpr const char *name = "Triangle Oscillator";
  static constexpr const char *description =
      "An anti-aliased triangle wave oscillator";
  static constexpr int numInputs = 2;
};

template <typename sample_type> struct OscillatorTraits<PulseOsc<sample_type>> {
  static constexpr const char *name = "Pulse Oscillator";
  static constexpr const char *description =
      "An anti-aliased pulse wave oscillator with pulse width control";
  static constexpr int numInputs = 3;
};

// Base Oscillator class
template <typename sample_type> class Oscillator : public Module<sample_type> {
public:
  virtual sample_type getCurrentValue() const = 0;

  void setFrequency(sample_type frequency) {
    assert(frequency > 0);
    m_frequency = frequency;
  }

  void setAmplitude(sample_type amplitude) {
    assert(amplitude >= 0);
    m_amplitude = amplitude;
  }

  void setPhase(sample_type phase) { m_phase = phase; }

  void setParameter(const std::string &name, sample_type value) override {
    if (name == "frequency") {
      setFrequency(value);
    } else if (name == "amplitude") {
      setAmplitude(value);
    } else if (name == "phase") {
      setPhase(value);
    } else {
      throw std::invalid_argument("Unknown parameter: " + name);
    }
  }

  sample_type getParameter(const std::string &name) const override {
    if (name == "frequency") {
      return getFrequency();
    }
    if (name == "amplitude") {
      return getAmplitude();
    }
    if (name == "phase") {
      return getPhase();
    }
    throw std::invalid_argument("Unknown parameter: " + name);
  }

  [[nodiscard]] std::vector<std::string> getParameterNames() const override {
    return {"frequency", "amplitude", "phase"};
  }

  [[nodiscard]] std::string getInputName(unsigned int index) const override {
    switch (index) {
    case 0:
      return "Frequency Modulation";
    case 1:
      return "Amplitude Modulation";
    default:
      return "";
    }
  }

  [[nodiscard]] std::string getOutputName(unsigned int index) const override {
    return (index == 0) ? "output" : "";
  }

  [[nodiscard]] unsigned int getNumOutputs() const override { return 1; }

  void reset() override { m_phase = 0; }

  void prepare(unsigned int sampleRate) override {
    AudioEngine::setSampleRate(sampleRate);
  }

  sample_type getFrequency() const { return m_frequency; }
  sample_type getAmplitude() const { return m_amplitude; }
  sample_type getPhase() const { return m_phase; }

  virtual ~Oscillator() = default;

  Oscillator(const Oscillator &) = delete;
  Oscillator(Oscillator &&) = delete;
  Oscillator &operator=(const Oscillator &) = delete;
  Oscillator &operator=(Oscillator &&) = delete;

protected:
  sample_type polyBlep(sample_type t, sample_type dt) const {
    if (t < dt) {
      t /= dt;
      return t + t - t * t - 1.0;
    }
    if (t > 1.0 - dt) {
      t = (t - 1.0) / dt;
      return t * t + t + t + 1.0;
    }
    return 0.0;
  }

  sample_type m_frequency{440.0};
  sample_type m_amplitude{1.0};
  sample_type m_phase{0.0};
};

// SIMD Oscillator base class
template <typename Derived, typename sample_type>
class SIMDOscillator : public Oscillator<sample_type> {
protected:
  using vec_t = llvm::SmallVector<sample_type, 8>;

  void applyModulation(vec_t &freqVec, vec_t &ampVec,
                       const SIMDProcessingParams<sample_type> &simdParams,
                       unsigned int frameIndex, int vectorSize) {
    if (simdParams.freqMod && *simdParams.freqMod) {
      for (int j = 0; j < vectorSize; ++j) {
        freqVec[j] += (*simdParams.freqMod)[frameIndex + j];
      }
    }
    if (simdParams.ampMod && *simdParams.ampMod) {
      for (int j = 0; j < vectorSize; ++j) {
        ampVec[j] *= (*simdParams.ampMod)[frameIndex + j];
      }
    }
  }

  void updatePhase(vec_t &phaseVec, const vec_t &phaseIncVec,
                   const vec_t &twoPi) {
    for (size_t j = 0; j < phaseVec.size(); ++j) {
      phaseVec[j] += phaseIncVec[j];
      if (phaseVec[j] >= twoPi[j]) {
        phaseVec[j] -= twoPi[j];
      }
    }
  }

  virtual vec_t generateWaveform(const vec_t &tVec, const vec_t &dtVec) = 0;

  void processSIMD(SIMDProcessingParams<sample_type> &simdParams,
                   OscillatorParams<sample_type> &oscParams) {
    constexpr int vectorSize = 8;
    const vec_t twoPi(vectorSize, Constants<sample_type>::twoPiConstant);
    const vec_t sampleRateVec(vectorSize, oscParams.sampleRate);

    vec_t phaseVec(vectorSize, oscParams.phase);
    vec_t freqVec(vectorSize, oscParams.baseFrequency);
    vec_t ampVec(vectorSize, oscParams.baseAmplitude);

    for (unsigned int i = 0; i < simdParams.numFrames; i += vectorSize) {
      applyModulation(freqVec, ampVec, simdParams, i, vectorSize);

      vec_t phaseIncVec = freqVec * twoPi / sampleRateVec;
      vec_t tVec = phaseVec / twoPi;
      vec_t dtVec = freqVec / sampleRateVec;

      vec_t waveform =
          static_cast<Derived *>(this)->generateWaveform(tVec, dtVec);
      vec_t outputVec = ampVec * waveform;

      for (int j = 0; j < vectorSize && i + j < simdParams.numFrames; ++j) {
        simdParams.output[i + j] = outputVec[j];
      }

      updatePhase(phaseVec, phaseIncVec, twoPi);
    }

    oscParams.phase = phaseVec.back();
  }

public:
  void process(const std::vector<std::optional<sample_type *>> &inputs,
               std::vector<sample_type *> &outputs,
               unsigned int numFrames) override {
    if (outputs.empty() || numFrames == 0)
      return;

    sample_type *output = outputs[0];
    std::optional<sample_type *> freqMod =
        !inputs.empty() ? inputs[0] : std::nullopt;
    std::optional<sample_type *> ampMod =
        inputs.size() > 1 ? inputs[1] : std::nullopt;

    OscillatorParams<sample_type> oscParams{
        this->getPhase(), this->getFrequency(), this->getAmplitude(),
        static_cast<sample_type>(AudioEngine::getSampleRate())};

    SIMDProcessingParams<sample_type> simdParams{output, freqMod, ampMod,
                                                 numFrames};

    processSIMD(simdParams, oscParams);

    this->setPhase(oscParams.phase);
  }

  [[nodiscard]] std::string getName() const override {
    return OscillatorTraits<Derived>::name;
  }

  [[nodiscard]] std::string getDescription() const override {
    return OscillatorTraits<Derived>::description;
  }

  [[nodiscard]] unsigned int getNumInputs() const override {
    return OscillatorTraits<Derived>::numInputs;
  }

  [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived &>(*this));
  }
};

// Sine Oscillator
template <typename sample_type>
class SineOsc : public SIMDOscillator<SineOsc<sample_type>, sample_type> {
protected:
  friend class SIMDOscillator<SineOsc<sample_type>, sample_type>;

  typename SIMDOscillator<SineOsc<sample_type>, sample_type>::vec_t
  generateWaveform(const typename SIMDOscillator<SineOsc<sample_type>,
                                                 sample_type>::vec_t &tVec,
                   const typename SIMDOscillator<SineOsc<sample_type>,
                                                 sample_type>::vec_t &) {
    typename SIMDOscillator<SineOsc<sample_type>, sample_type>::vec_t result(
        tVec.size());
    for (size_t i = 0; i < tVec.size(); ++i) {
      result[i] = std::sin(tVec[i] * Constants<sample_type>::twoPiConstant);
    }
    return result;
  }

public:
  sample_type getCurrentValue() const override {
    return this->getAmplitude() * std::sin(this->getPhase());
  }
};

// Sawtooth Oscillator
template <typename sample_type>
class SawOsc : public SIMDOscillator<SawOsc<sample_type>, sample_type> {
protected:
  friend class SIMDOscillator<SawOsc<sample_type>, sample_type>;

  typename SIMDOscillator<SawOsc<sample_type>, sample_type>::vec_t
  generateWaveform(const typename SIMDOscillator<SawOsc<sample_type>,
                                                 sample_type>::vec_t &tVec,
                   const typename SIMDOscillator<SawOsc<sample_type>,
                                                 sample_type>::vec_t &dtVec) {
    typename SIMDOscillator<SawOsc<sample_type>, sample_type>::vec_t result(
        tVec.size());
    for (size_t i = 0; i < tVec.size(); ++i) {
      result[i] = 2.0 * tVec[i] - 1.0 - this->polyBlep(tVec[i], dtVec[i]);
    }
    return result;
  }

public:
  sample_type getCurrentValue() const override {
    sample_type t = this->getPhase() / Constants<sample_type>::twoPiConstant;
    sample_type dt = this->getFrequency() /
                     static_cast<sample_type>(AudioEngine::getSampleRate());
    return this->getAmplitude() * (2.0 * t - 1.0 - this->polyBlep(t, dt));
  }
};

// Triangle Oscillator
template <typename sample_type>
class TriangleOsc
    : public SIMDOscillator<TriangleOsc<sample_type>, sample_type> {
protected:
  friend class SIMDOscillator<TriangleOsc<sample_type>, sample_type>;

  typename SIMDOscillator<TriangleOsc<sample_type>, sample_type>::vec_t
  generateWaveform(const typename SIMDOscillator<TriangleOsc<sample_type>,
                                                 sample_type>::vec_t &tVec,
                   const typename SIMDOscillator<TriangleOsc<sample_type>,
                                                 sample_type>::vec_t &dtVec) {
    typename SIMDOscillator<TriangleOsc<sample_type>, sample_type>::vec_t
        result(tVec.size());
    for (size_t i = 0; i < tVec.size(); ++i) {
      sample_type t = tVec[i];
      sample_type dt = dtVec[i];
      sample_type value = 2.0 * std::abs(2.0 * t - 1.0) - 1.0;
      sample_type blep1 = this->polyBlep(t, dt);
      sample_type blep2 = this->polyBlep(std::fmod(t + 0.5, 1.0), dt);
      result[i] = value + 4.0 * (blep1 - blep2);
    }
    return result;
  }

public:
  sample_type getCurrentValue() const override {
    sample_type t = this->getPhase() / Constants<sample_type>::twoPiConstant;
    sample_type dt = this->getFrequency() /
                     static_cast<sample_type>(AudioEngine::getSampleRate());
    sample_type value = 2.0 * std::abs(2.0 * t - 1.0) - 1.0;
    sample_type blep1 = this->polyBlep(t, dt);
    sample_type blep2 = this->polyBlep(std::fmod(t + 0.5, 1.0), dt);
    return this->getAmplitude() * (value + 4.0 * (blep1 - blep2));
  }
};

// Pulse Oscillator
template <typename sample_type>
class PulseOsc : public SIMDOscillator<PulseOsc<sample_type>, sample_type> {
protected:
  friend class SIMDOscillator<PulseOsc<sample_type>, sample_type>;

  typename SIMDOscillator<PulseOsc<sample_type>, sample_type>::vec_t
  generateWaveform(const typename SIMDOscillator<PulseOsc<sample_type>,
                                                 sample_type>::vec_t &tVec,
                   const typename SIMDOscillator<PulseOsc<sample_type>,
                                                 sample_type>::vec_t &dtVec) {
    typename SIMDOscillator<PulseOsc<sample_type>, sample_type>::vec_t result(
        tVec.size());
    for (size_t i = 0; i < tVec.size(); ++i) {
      sample_type t = tVec[i];
      sample_type dt = dtVec[i];
      sample_type value = (t < m_pulseWidth) ? 1.0 : -1.0;
      value -= this->polyBlep(t, dt);
      value += this->polyBlep(std::fmod(t + 1.0 - m_pulseWidth, 1.0), dt);
      result[i] = value;
    }
    return result;
  }

public:
  PulseOsc() : m_pulseWidth(0.5) {}

  sample_type getCurrentValue() const override {
    sample_type t = this->getPhase() / Constants<sample_type>::twoPiConstant;
    sample_type dt = this->getFrequency() /
                     static_cast<sample_type>(AudioEngine::getSampleRate());
    sample_type value = (t < m_pulseWidth) ? 1.0 : -1.0;
    value -= this->polyBlep(t, dt);
    value += this->polyBlep(std::fmod(t + 1.0 - m_pulseWidth, 1.0), dt);
    return this->getAmplitude() * value;
  }

  void setPulseWidth(sample_type pulseWidth) {
    m_pulseWidth = std::clamp(pulseWidth, static_cast<sample_type>(0.0),
                              static_cast<sample_type>(1.0));
  }

  sample_type getPulseWidth() const { return m_pulseWidth; }

  void setParameter(const std::string &name, sample_type value) override {
    if (name == "pulseWidth") {
      setPulseWidth(value);
    } else {
      SIMDOscillator<PulseOsc<sample_type>, sample_type>::setParameter(name,
                                                                       value);
    }
  }

  sample_type getParameter(const std::string &name) const override {
    if (name == "pulseWidth") {
      return getPulseWidth();
    }
    return SIMDOscillator<PulseOsc<sample_type>, sample_type>::getParameter(
        name);
  }

  [[nodiscard]] std::vector<std::string> getParameterNames() const override {
    auto names =
        SIMDOscillator<PulseOsc<sample_type>, sample_type>::getParameterNames();
    names.push_back("pulseWidth");
    return names;
  }

  void process(const std::vector<std::optional<sample_type *>> &inputs,
               std::vector<sample_type *> &outputs,
               unsigned int numFrames) override {
    if (outputs.empty() || numFrames == 0)
      return;

    sample_type *output = outputs[0];
    std::optional<sample_type *> freqMod =
        !inputs.empty() ? inputs[0] : std::nullopt;
    std::optional<sample_type *> ampMod =
        inputs.size() > 1 ? inputs[1] : std::nullopt;
    std::optional<sample_type *> pwMod =
        inputs.size() > 2 ? inputs[2] : std::nullopt;

    OscillatorParams<sample_type> oscParams{
        this->getPhase(), this->getFrequency(), this->getAmplitude(),
        static_cast<sample_type>(AudioEngine::getSampleRate())};

    SIMDProcessingParams<sample_type> simdParams{output, freqMod, ampMod, pwMod,
                                                 numFrames};

    this->processSIMD(simdParams, oscParams);

    this->setPhase(oscParams.phase);
  }

private:
  sample_type m_pulseWidth;
};

} // namespace tinysynth
