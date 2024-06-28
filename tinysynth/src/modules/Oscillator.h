#ifndef OSCILLATOR_H
#define OSCILLATOR_H

#include "../core/Module.h"
#include "../utils/Utils.h"
#include "../utils/Constants.h"
#include "AudioEngine.h"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <immintrin.h>

namespace tinysynth {

template <typename sample_type>
class Oscillator : public Module<sample_type> {
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

    void setPhase(sample_type phase) {
        m_phase = phase;
    }

    void setParameter(const std::string& name, sample_type value) override {
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

    sample_type getParameter(const std::string& name) const override {
        if (name == "frequency") {
            return getFrequency();
        } else if (name == "amplitude") {
            return getAmplitude();
        } else if (name == "phase") {
            return getPhase();
        } else {
            throw std::invalid_argument("Unknown parameter: " + name);
        }
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

    [[nodiscard]] unsigned int getNumOutputs() const override {
        return 1;
    }

    void reset() override {
        m_phase = 0;
    }

    void prepare(unsigned int sampleRate) override {
        AudioEngine::setSampleRate(sampleRate);
    }

    sample_type getFrequency() const {
        return m_frequency;
    }

    sample_type getAmplitude() const {
        return m_amplitude;
    }

    sample_type getPhase() const {
        return m_phase;
    }

    virtual ~Oscillator() = default;

    Oscillator(const Oscillator&) = delete;
    Oscillator(Oscillator&&) = delete;
    Oscillator& operator=(const Oscillator&) = delete;
    Oscillator& operator=(Oscillator&&) = delete;

private:
    sample_type m_frequency{440.0};
    sample_type m_amplitude{1.0};
    sample_type m_phase{0.0};
};

template <typename sample_type>
class SineOsc : public Oscillator<sample_type> {
public:
    void process(const std::vector<sample_type*>& inputs,
                 std::vector<sample_type*>& outputs,
                 unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0) return;

        sample_type* output = outputs[0];
        sample_type* freqMod = (inputs.size() > 0) ? inputs[0] : nullptr;
        sample_type* ampMod = (inputs.size() > 1) ? inputs[1] : nullptr;

        sample_type phase = this->getPhase();
        sample_type baseFrequency = this->getFrequency();
        sample_type baseAmplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(AudioEngine::getSampleRate());
        const sample_type phaseIncrement = baseFrequency * Constants<sample_type>::twoPiConstant / sampleRate;

        for (unsigned int i = 0; i < numFrames; ++i) {
            sample_type currentFrequency = baseFrequency;
            sample_type currentAmplitude = baseAmplitude;

            if (freqMod) {
                currentFrequency += freqMod[i];
            }
            if (ampMod) {
                currentAmplitude *= ampMod[i];
            }

            output[i] = currentAmplitude * std::sin(phase);
            phase += currentFrequency * Constants<sample_type>::twoPiConstant / sampleRate;
            if (phase >= Constants<sample_type>::twoPiConstant) {
                phase -= Constants<sample_type>::twoPiConstant;
            }
        }

        this->setPhase(phase);
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
};

template <typename sample_type>
class SawOsc : public Oscillator<sample_type> {
public:
    void process(const std::vector<sample_type*>& inputs,
                 std::vector<sample_type*>& outputs,
                 unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0) return;

        sample_type* output = outputs[0];
        sample_type* freqMod = (inputs.size() > 0) ? inputs[0] : nullptr;
        sample_type* ampMod = (inputs.size() > 1) ? inputs[1] : nullptr;

        sample_type phase = this->getPhase();
        sample_type baseFrequency = this->getFrequency();
        sample_type baseAmplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(AudioEngine::getSampleRate());
        const sample_type phaseIncrement = baseFrequency * Constants<sample_type>::twoPiConstant / sampleRate;

        for (unsigned int i = 0; i < numFrames; ++i) {
            sample_type currentFrequency = baseFrequency;
            sample_type currentAmplitude = baseAmplitude;

            if (freqMod) {
                currentFrequency += freqMod[i];
            }
            if (ampMod) {
                currentAmplitude *= ampMod[i];
            }

            output[i] = currentAmplitude * ((phase / Constants<sample_type>::piConstant) - 1);
            phase += currentFrequency * Constants<sample_type>::twoPiConstant / sampleRate;
            if (phase >= Constants<sample_type>::twoPiConstant) {
                phase -= Constants<sample_type>::twoPiConstant;
            }
        }

        this->setPhase(phase);
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

    sample_type getCurrentValue() const override {
        return this->getAmplitude() * ((this->getPhase() / Constants<sample_type>::piConstant) - 1);
    }
};

template <typename sample_type>
class TriangleOsc : public Oscillator<sample_type> {
public:
    void process(const std::vector<sample_type*>& inputs,
                 std::vector<sample_type*>& outputs,
                 unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0) return;

        sample_type* output = outputs[0];
        sample_type* freqMod = (inputs.size() > 0) ? inputs[0] : nullptr;
        sample_type* ampMod = (inputs.size() > 1) ? inputs[1] : nullptr;

        sample_type phase = this->getPhase();
        sample_type baseFrequency = this->getFrequency();
        sample_type baseAmplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(AudioEngine::getSampleRate());
        const sample_type phaseIncrement = baseFrequency * Constants<sample_type>::twoPiConstant / sampleRate;

        for (unsigned int i = 0; i < numFrames; ++i) {
            sample_type currentFrequency = baseFrequency;
            sample_type currentAmplitude = baseAmplitude;

            if (freqMod) {
                currentFrequency += freqMod[i];
            }
            if (ampMod) {
                currentAmplitude *= ampMod[i];
            }

            output[i] = currentAmplitude * ((phase < Constants<sample_type>::piConstant) ? -1 + (2 * phase / Constants<sample_type>::piConstant)
                                                        : 3 - (2 * phase / Constants<sample_type>::piConstant));
            phase += currentFrequency * Constants<sample_type>::twoPiConstant / sampleRate;
            if (phase >= Constants<sample_type>::twoPiConstant) {
                phase -= Constants<sample_type>::twoPiConstant;
            }
        }

        this->setPhase(phase);
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

    sample_type getCurrentValue() const override {
        const sample_type phase = this->getPhase();
        return this->getAmplitude() * ((phase < Constants<sample_type>::piConstant) ? -1 + (2 * phase / Constants<sample_type>::piConstant)
                                                          : 3 - (2 * phase / Constants<sample_type>::piConstant));
    }
};

} // namespace tinysynth

#endif // OSCILLATOR_H
