#ifndef OSCILLATOR_H
#define OSCILLATOR_H

#include "../core/Module.h"
#include "../utils/Utils.h"
#include "AudioEngine.h"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinysynth {

template <typename sample_type> class Oscillator : public Module<sample_type> {
  public:
    static constexpr sample_type PI = static_cast<sample_type>(3.14159265358979323846);
    static constexpr sample_type TWO_PI = 2 * PI;

    Oscillator() : m_frequency(440.0), m_phase(0), m_amplitude(1) {}

    [[nodiscard]] unsigned int getNumInputs() const override { return 0; }
    [[nodiscard]] unsigned int getNumOutputs() const override { return 1; }

    [[nodiscard]] std::string
    getInputName([[maybe_unused]] unsigned int index) const override {
        return "";
    }

    [[nodiscard]] std::string getOutputName(unsigned int index) const override {
        return (index == 0) ? "Output" : throw std::out_of_range("Invalid output index");
    }

    void setParameter(const std::string &name, sample_type value) override {
        if (name == "frequency") {
            setFrequency(value);
        } else if (name == "amplitude") {
            setAmplitude(value);
        }
    }

    [[nodiscard]] sample_type getParameter(const std::string &name) const override {
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

    void prepare(unsigned int sampleRate) override {
        AudioEngine::setSampleRate(sampleRate);
    }

    void setFrequency(sample_type freq) {
        m_frequency = util::clamp(freq, static_cast<sample_type>(20),
                                  static_cast<sample_type>(20000));
    }

    sample_type getFrequency() const { return m_frequency; }

    void setAmplitude(sample_type amp) {
        m_amplitude =
            util::clamp(amp, static_cast<sample_type>(0), static_cast<sample_type>(1));
    }

    sample_type getAmplitude() const { return m_amplitude; }

    virtual sample_type getCurrentValue() const = 0;

    void setPhase(sample_type phase) {
        m_phase = std::fmod(phase, TWO_PI);
        if (m_phase < 0) {
            m_phase += TWO_PI;
        }
    }

    sample_type getPhase() const { return m_phase; }

    [[nodiscard]] unsigned int getSampleRate() const {
        return AudioEngine::getSampleRate();
    }

    virtual ~Oscillator() = default;

    Oscillator(const Oscillator &) = delete;
    Oscillator(Oscillator &&) = delete;
    Oscillator &operator=(const Oscillator &) = delete;
    Oscillator &operator=(Oscillator &&) = delete;

  private:
    sample_type m_frequency;
    sample_type m_phase;
    sample_type m_amplitude;
};

template <typename sample_type> class SinOsc : public Oscillator<sample_type> {
  public:
    void process(std::vector<sample_type *> &outputs, unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0) {
            return;
        }

        sample_type *output = outputs[0];
        sample_type phase = this->getPhase();
        const sample_type frequency = this->getFrequency();
        const sample_type amplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(this->getSampleRate());
        const sample_type phaseIncrement = frequency * this->TWO_PI / sampleRate;

        for (unsigned int i = 0; i < numFrames; ++i) {
            output[i] = amplitude * std::sin(phase);
            phase += phaseIncrement;
            if (phase >= this->TWO_PI) {
                phase -= this->TWO_PI;
            }
        }

        this->setPhase(phase);
    }

    [[nodiscard]] std::string getName() const override { return "Sine Oscillator"; }

    [[nodiscard]] std::string getDescription() const override {
        return "A sine wave oscillator";
    }

    [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
        return std::make_unique<SinOsc>(*this);
    }

    sample_type getCurrentValue() const override {
        return this->getAmplitude() * std::sin(this->getPhase());
    }
};

template <typename sample_type> class SquareOsc : public Oscillator<sample_type> {
  public:
    SquareOsc() : m_pulseWidth(0.5) {}

    void process(std::vector<sample_type *> &outputs, unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0) {
            return;
}

        sample_type *output = outputs[0];
        sample_type phase = this->getPhase();
        const sample_type frequency = this->getFrequency();
        const sample_type amplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(this->getSampleRate());
        const sample_type phaseIncrement = frequency * this->TWO_PI / sampleRate;

        for (unsigned int i = 0; i < numFrames; ++i) {
            output[i] = amplitude * ((phase < this->TWO_PI * m_pulseWidth) ? 1 : -1);
            phase += phaseIncrement;
            if (phase >= this->TWO_PI) {
                phase -= this->TWO_PI;
            }
        }

        this->setPhase(phase);
    }

    [[nodiscard]] std::string getName() const override { return "Square Oscillator"; }

    [[nodiscard]] std::string getDescription() const override {
        return "A square wave oscillator";
    }

    [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
        return std::make_unique<SquareOsc>(*this);
    }

    void setParameter(const std::string &name, sample_type value) override {
        Oscillator<sample_type>::setParameter(name, value);
        if (name == "pulseWidth") {
            m_pulseWidth = util::clamp(value, static_cast<sample_type>(0),
                                       static_cast<sample_type>(1));
        }
    }

    [[nodiscard]] sample_type getParameter(const std::string &name) const override {
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

    sample_type getCurrentValue() const override {
        return this->getAmplitude() *
               ((this->getPhase() < this->TWO_PI * m_pulseWidth) ? 1 : -1);
    }

  private:
    sample_type m_pulseWidth;
};

template <typename sample_type> class SawOsc : public Oscillator<sample_type> {
  public:
    void process(std::vector<sample_type *> &outputs, unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0) {
            return;
}

        sample_type *output = outputs[0];
        sample_type phase = this->getPhase();
        const sample_type frequency = this->getFrequency();
        const sample_type amplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(this->getSampleRate());
        const sample_type phaseIncrement = frequency * this->TWO_PI / sampleRate;

        for (unsigned int i = 0; i < numFrames; ++i) {
            output[i] = amplitude * ((phase / this->PI) - 1);
            phase += phaseIncrement;
            if (phase >= this->TWO_PI) {
                phase -= this->TWO_PI;
            }
        }

        this->setPhase(phase);
    }

    [[nodiscard]] std::string getName() const override { return "Sawtooth Oscillator"; }

    [[nodiscard]] std::string getDescription() const override {
        return "A sawtooth wave oscillator";
    }

    [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
        return std::make_unique<SawOsc>(*this);
    }

    sample_type getCurrentValue() const override {
        return this->getAmplitude() * ((this->getPhase() / this->PI) - 1);
    }
};

template <typename sample_type> class TriangleOsc : public Oscillator<sample_type> {
  public:
    void process(std::vector<sample_type *> &outputs, unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0) {
            return;
}

        sample_type *output = outputs[0];
        sample_type phase = this->getPhase();
        const sample_type frequency = this->getFrequency();
        const sample_type amplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(this->getSampleRate());
        const sample_type phaseIncrement = frequency * this->TWO_PI / sampleRate;

        for (unsigned int i = 0; i < numFrames; ++i) {
            output[i] = amplitude * ((phase < this->PI) ? -1 + (2 * phase / this->PI)
                                                        : 3 - (2 * phase / this->PI));
            phase += phaseIncrement;
            if (phase >= this->TWO_PI) {
                phase -= this->TWO_PI;
            }
        }

        this->setPhase(phase);
    }

    [[nodiscard]] std::string getName() const override { return "Triangle Oscillator"; }

    [[nodiscard]] std::string getDescription() const override {
        return "A triangle wave oscillator";
    }

    [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
        return std::make_unique<TriangleOsc>(*this);
    }

    sample_type getCurrentValue() const override {
        const sample_type phase = this->getPhase();
        return this->getAmplitude() * ((phase < this->PI) ? -1 + (2 * phase / this->PI)
                                                          : 3 - (2 * phase / this->PI));
    }
};

} // namespace tinysynth

#endif // OSCILLATOR_H