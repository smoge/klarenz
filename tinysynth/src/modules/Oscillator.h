#ifndef OSCILLATOR_H
#define OSCILLATOR_H

#include "../core/Module.h"
#include "../utils/Utils.h"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinysynth {

template <typename sample_type> class Oscillator : public Module<sample_type> {
public:
  static constexpr sample_type PI =
      static_cast<sample_type>(3.14159265358979323846);
  static constexpr sample_type TWO_PI = 2 * PI;

  Oscillator()
      : m_frequency(440.0), m_phase(0), m_amplitude(1), m_sampleRate(44100) {}

  [[nodiscard]] unsigned int getNumInputs() const override { return 0; }
  [[nodiscard]] unsigned int getNumOutputs() const override { return 1; }
  [[nodiscard]] std::string getInputName(unsigned int index) const override {
    throw std::out_of_range("Oscillator has no inputs");
  }
  [[nodiscard]] std::string getOutputName(unsigned int index) const override {
    return (index == 0) ? "Output"
                        : throw std::out_of_range("Invalid output index");
  }

  void setParameter(const std::string &name, sample_type value) override {
    if (name == "frequency") {
      m_frequency = util::clamp(value, static_cast<sample_type>(20),
                                static_cast<sample_type>(20000));
    } else if (name == "amplitude") {
      m_amplitude = clamp(value, static_cast<sample_type>(0),
                          static_cast<sample_type>(1));
    }
  }

  [[nodiscard]] sample_type
  getParameter(const std::string &name) const override {
    if (name == "frequency") {
      return m_frequency;
    }
    if (name == "amplitude") {
      return m_amplitude;
    }
    throw std::out_of_range("Invalid parameter name");
  }

  [[nodiscard]] std::vector<std::string> getParameterNames() const override {
    return {"frequency", "amplitude"};
  }


  void reset() override { m_phase = 0; }
  void prepare(unsigned int sampleRate) override { m_sampleRate = sampleRate; }

  void setFrequency(sample_type freq) { m_frequency = clamp(freq, 20, 20000); }
  sample_type getFrequency() const { return m_frequency; }

  void setAmplitude(sample_type amp) { m_amplitude = clamp(amp, 0, 1); }
  sample_type getAmplitude() const { return m_amplitude; }

  virtual sample_type getCurrentValue() const = 0;

  void setPhase(sample_type phase) {
    m_phase = std::fmod(phase, TWO_PI);
    if (m_phase < 0) {
      m_phase += TWO_PI;
    }
  }

  void setSampleRate(unsigned int rate) { m_sampleRate = rate; }
  [[nodiscard]] unsigned int getSampleRate() const { return m_sampleRate; }

  // Phase is typically not directly set, but you might want a getter
  sample_type getPhase() const { return m_phase; }

  virtual ~Oscillator() = default;

private:
  sample_type m_frequency;
  sample_type m_phase;
  sample_type m_amplitude;
  unsigned int m_sampleRate;
};

template <typename sample_type> class SinOsc : public Oscillator<sample_type> {
public:
  void process(std::vector<sample_type *> &outputs,
               unsigned int numFrames) override {
    if (outputs.empty() || numFrames == 0) {
      return;
    }
    sample_type *output = outputs[0];
    
    sample_type phaseIncrement = this->getFrequency() * this->TWO_PI /
                             static_cast<sample_type>(this->getSampleRate());

    for (unsigned int i = 0; i < numFrames; ++i) {
      output[i] = this->getAmplitude * std::sin(this->getPhase);
      this->getPhase += phaseIncrement;
      if (this->getPhase >= this->TWO_PI) {
        this->getPhase -= this->TWO_PI;
      }
    }
  }

  [[nodiscard]] std::string getName() const override {
    return "Sine Oscillator";
  }
  [[nodiscard]] std::string getDescription() const override {
    return "A sine wave oscillator";
  }
  [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
    return std::make_unique<SinOsc>(*this);
  }
};

template <typename sample_type>
class SquareOsc : public Oscillator<sample_type> {
public:
  SquareOsc() : m_pulseWidth(0.5) {}

  void process(std::vector<sample_type *> &outputs,
               unsigned int numFrames) override {
    if (outputs.empty() || numFrames == 0) {
      return;
    }
    sample_type *output = outputs[0];
    sample_type phaseIncrement = this->m_frequency * this->TWO_PI /
                                 static_cast<sample_type>(this->m_sampleRate);

    for (unsigned int i = 0; i < numFrames; ++i) {
      output[i] = this->getAmplitude *
                  ((this->getPhase < this->TWO_PI * m_pulseWidth) ? 1 : -1);
      this->getPhase += phaseIncrement;
      if (this->getPhase >= this->TWO_PI) {
        this->getPhase -= this->TWO_PI;
      }
    }
  }

  [[nodiscard]] std::string getName() const override {
    return "Square Oscillator";
  }
  [[nodiscard]] std::string getDescription() const override {
    return "A square wave oscillator";
  }
  [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
    return std::make_unique<SquareOsc>(*this);
  }

  void setParameter(const std::string &name, sample_type value) override {
    Oscillator<sample_type>::setParameter(name, value);
    if (name == "pulseWidth") {
      m_pulseWidth = this->template clamp<sample_type>(value, 0, 1);
    }
  }

  [[nodiscard]] sample_type
  getParameter(const std::string &name) const override {
    if (name == "pulseWidth") {
      return m_pulseWidth;
    }
    return Oscillator<sample_type>::getParameter(name);
  }

  [[nodiscard]] std::vector<std::string> getParameterNames() const override {
    auto names = Oscillator<sample_type>::getParameterNames();
    names.push_back("pulseWidth");
    return names;
  }

private:
  sample_type m_pulseWidth;
};

// SawOsc.h
template <typename sample_type> class SawOsc : public Oscillator<sample_type> {
public:
  void process(std::vector<sample_type *> &outputs,
               unsigned int numFrames) override {
    if (outputs.empty() || numFrames == 0) {
      return;
    }
    sample_type *output = outputs[0];
    sample_type phaseIncrement = this->m_frequency * this->TWO_PI /
                                 static_cast<sample_type>(this->getSampleRate);

    for (unsigned int i = 0; i < numFrames; ++i) {
      output[i] = this->getAmplitude * ((this->getPhase / this->PI) - 1);
      this->getPhase += phaseIncrement;
      if (this->getPhase >= this->TWO_PI) {
        this->getPhase -= this->TWO_PI;
      }
    }
  }

  [[nodiscard]] std::string getName() const override {
    return "Sawtooth Oscillator";
  }
  [[nodiscard]] std::string getDescription() const override {
    return "A sawtooth wave oscillator";
  }
  [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
    return std::make_unique<SawOsc>(*this);
  }
};

// TriangleOsc.h
template <typename sample_type>
class TriangleOsc : public Oscillator<sample_type> {
public:
  void process(std::vector<sample_type *> &outputs,
               unsigned int numFrames) override {
    if (outputs.empty() || numFrames == 0) {
      return;
    }
    sample_type *output = outputs[0];
    sample_type phaseIncrement = this->m_frequency * this->TWO_PI /
                                 static_cast<sample_type>(this->getSampleRate);

    for (unsigned int i = 0; i < numFrames; ++i) {
      output[i] =
          this->getAmplitude * ((this->getPhase < this->PI)
                                    ? -1 + (2 * this->getPhase / this->PI)
                                    : 3 - (2 * this->getPhase / this->PI));
      this->getPhase += phaseIncrement;
      if (this->getPhase >= this->TWO_PI) {
        this->getPhase -= this->TWO_PI;
      }
    }
  }

  [[nodiscard]] std::string getName() const override {
    return "Triangle Oscillator";
  }
  [[nodiscard]] std::string getDescription() const override {
    return "A triangle wave oscillator";
  }
  [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
    return std::make_unique<TriangleOsc>(*this);
  }
};

} // namespace tinysynth

#endif // OSCILLATOR_H