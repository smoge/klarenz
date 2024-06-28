
#pragma once

#include "../core/Module.h"
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

  vec_t polyBlepSIMD(const vec_t &t, const vec_t &dt) {
    vec_t result(t.size(), 0.0);
    for (size_t i = 0; i < t.size(); ++i) {
      if (t[i] < dt[i]) {
        sample_type t1 = t[i] / dt[i];
        result[i] = t1 + t1 - t1 * t1 - 1.0;
      } else if (t[i] > 1.0 - dt[i]) {
        sample_type t2 = (t[i] - 1.0) / dt[i];
        result[i] = t2 * t2 + t2 + t2 + 1.0;
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
