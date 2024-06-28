#ifndef OSCILLATOR_LLVM_H
#define OSCILLATOR_LLVM_H

#include "../core/Module.h"
#include "../utils/Constants.h"
#include "../utils/Utils.h"
#include "AudioEngine.h"
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/VectorType.h>
#include <llvm/Support/MathExtras.h>

// This file contains the implementation of the Oscillator class and its derived classes
// With LLVM SIMD optimizations and anti-aliasing techniques

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

template <typename sample_type> class SineOsc : public Module<sample_type> {
  public:
    void process(const std::vector<std::optional<sample_type *>> &inputs,
                 std::vector<sample_type *> &outputs, unsigned int numFrames) override {

        // void process(const std::vector<std::optional<sample_type *>> &inputs,
        //              std::vector<sample_type *> &outputs, unsigned int numFrames)
        //              override {

        if (outputs.empty() || numFrames == 0) {
            return;
        }

        sample_type *output = outputs[0];
        std::optional<sample_type *> freqMod = !inputs.empty() ? inputs[0] : std::nullopt;
        std::optional<sample_type *> ampMod =
            inputs.size() > 1 ? inputs[1] : std::nullopt;

        sample_type phase = this->getPhase();
        sample_type baseFrequency = this->getFrequency();
        sample_type baseAmplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(AudioEngine::getSampleRate());

        processSIMD(output, freqMod, ampMod, phase, baseFrequency, baseAmplitude,
                    sampleRate, numFrames);

        this->setPhase(phase);
    }
    [[nodiscard]] std::string getName() const override { return "Sine Oscillator"; }

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

    void processSIMD(sample_type *output, const std::optional<sample_type *> &freqMod,
                     const std::optional<sample_type *> &ampMod, sample_type &phase,
                     sample_type baseFrequency, sample_type baseAmplitude,
                     sample_type sampleRate, unsigned int numFrames) {
        constexpr int vectorSize = 8;
        const vec_t twoPi(vectorSize, Constants<sample_type>::twoPiConstant);
        const vec_t sampleRateVec(vectorSize, sampleRate);

        vec_t phaseVec(vectorSize, phase);
        vec_t freqVec(vectorSize, baseFrequency);
        vec_t ampVec(vectorSize, baseAmplitude);

        for (unsigned int i = 0; i < numFrames; i += vectorSize) {
            // Apply modulation
            if (freqMod && *freqMod) {
                for (int j = 0; j < vectorSize; ++j) {
                    freqVec[j] += (*freqMod)[i + j];
                }
            }
            if (ampMod && *ampMod) {
                for (int j = 0; j < vectorSize; ++j) {
                    ampVec[j] *= (*ampMod)[i + j];
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
            for (int j = 0; j < vectorSize && i + j < numFrames; ++j) {
                output[i + j] = outputVec[j];
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
        phase = phaseVec.back();
    }
};

/////// SawOsc class definition ///////

template <typename sample_type> class SawOsc : public Module<sample_type> {
  public:
    void process(const std::vector<std::optional<sample_type *>> &inputs,
                 std::vector<sample_type *> &outputs, unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0)
            return;

        sample_type *output = outputs[0];
        std::optional<sample_type *> freqMod =
            inputs.size() > 0 ? inputs[0] : std::nullopt;
        std::optional<sample_type *> ampMod =
            inputs.size() > 1 ? inputs[1] : std::nullopt;

        sample_type phase = this->getPhase();
        sample_type baseFrequency = this->getFrequency();
        sample_type baseAmplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(AudioEngine::getSampleRate());

        processSIMD(output, freqMod, ampMod, phase, baseFrequency, baseAmplitude,
                    sampleRate, numFrames);

        this->setPhase(phase);
    }

    [[nodiscard]] std::string getName() const override { return "Sawtooth Oscillator"; }

    [[nodiscard]] std::string getDescription() const override {
        return "An anti-aliased sawtooth wave oscillator with LLVM SIMD optimization";
    }

    [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
        return std::make_unique<SawOsc>(*this);
    }

    sample_type getCurrentValue() const override {
        sample_type t = this->getPhase() / Constants<sample_type>::twoPiConstant;
        sample_type dt =
            this->getFrequency() / static_cast<sample_type>(AudioEngine::getSampleRate());
        return this->getAmplitude() * (2.0 * t - 1.0 - this->polyBlep(t, dt));
    }

  private:
    using vec_t = llvm::SmallVector<sample_type, 8>;

    void processSIMD(sample_type *output, const std::optional<sample_type *> &freqMod,
                     const std::optional<sample_type *> &ampMod, sample_type &phase,
                     sample_type baseFrequency, sample_type baseAmplitude,
                     sample_type sampleRate, unsigned int numFrames) {
        constexpr int vectorSize = 8;
        const vec_t twoPi(vectorSize, Constants<sample_type>::twoPiConstant);
        const vec_t sampleRateVec(vectorSize, sampleRate);
        const vec_t two(vectorSize, 2.0);
        const vec_t one(vectorSize, 1.0);

        vec_t phaseVec(vectorSize, phase);
        vec_t freqVec(vectorSize, baseFrequency);
        vec_t ampVec(vectorSize, baseAmplitude);

        for (unsigned int i = 0; i < numFrames; i += vectorSize) {
            // Apply modulation
            if (freqMod && *freqMod) {
                for (int j = 0; j < vectorSize; ++j) {
                    freqVec[j] += (*freqMod)[i + j];
                }
            }
            if (ampMod && *ampMod) {
                for (int j = 0; j < vectorSize; ++j) {
                    ampVec[j] *= (*ampMod)[i + j];
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
            for (int j = 0; j < vectorSize && i + j < numFrames; ++j) {
                output[i + j] = outputVec[j];
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
        phase = phaseVec.back();
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

template <typename sample_type> class TriangleOsc : public Module<sample_type> {
  public:
    void process(const std::vector<std::optional<sample_type *>> &inputs,
                 std::vector<sample_type *> &outputs, unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0) {
            return;
        }

        sample_type *output = outputs[0];
        std::optional<sample_type *> freqMod = !inputs.empty() ? inputs[0] : std::nullopt;
        std::optional<sample_type *> ampMod =
            inputs.size() > 1 ? inputs[1] : std::nullopt;

        sample_type phase = this->getPhase();
        sample_type baseFrequency = this->getFrequency();
        sample_type baseAmplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(AudioEngine::getSampleRate());

        processSIMD(output, freqMod, ampMod, phase, baseFrequency, baseAmplitude,
                    sampleRate, numFrames);

        this->setPhase(phase);
    }

    [[nodiscard]] std::string getName() const override { return "Triangle Oscillator"; }

    [[nodiscard]] std::string getDescription() const override {
        return "An anti-aliased triangle wave oscillator";
    }

    [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
        return std::make_unique<TriangleOsc>(*this);
    }

    sample_type getCurrentValue() const override {
        sample_type t = this->getPhase() / Constants<sample_type>::twoPiConstant;
        sample_type dt =
            this->getFrequency() / static_cast<sample_type>(AudioEngine::getSampleRate());

        sample_type value = 2.0 * std::abs(2.0 * t - 1.0) - 1.0;
        sample_type blep1 = this->polyBlep(t, dt);
        sample_type blep2 = this->polyBlep(std::fmod(t + 0.5, 1.0), dt);

        return this->getAmplitude() * (value + 4.0 * (blep1 - blep2));
    }

  private:
    using vec_t = llvm::SmallVector<sample_type, 8>;

    void processSIMD(sample_type *output, const std::optional<sample_type *> &freqMod,
                     const std::optional<sample_type *> &ampMod, sample_type &phase,
                     sample_type baseFrequency, sample_type baseAmplitude,
                     sample_type sampleRate, unsigned int numFrames) {
        constexpr int vectorSize = 8;
        const vec_t twoPi(vectorSize, Constants<sample_type>::twoPiConstant);
        const vec_t sampleRateVec(vectorSize, sampleRate);
        const vec_t two(vectorSize, 2.0);
        const vec_t one(vectorSize, 1.0);
        const vec_t four(vectorSize, 4.0);
        const vec_t half(vectorSize, 0.5);

        vec_t phaseVec(vectorSize, phase);
        vec_t freqVec(vectorSize, baseFrequency);
        vec_t ampVec(vectorSize, baseAmplitude);

        for (unsigned int i = 0; i < numFrames; i += vectorSize) {
            // Apply modulation
            if (freqMod && *freqMod) {
                for (int j = 0; j < vectorSize; ++j) {
                    freqVec[j] += (*freqMod)[i + j];
                }
            }
            if (ampMod && *ampMod) {
                for (int j = 0; j < vectorSize; ++j) {
                    ampVec[j] *= (*ampMod)[i + j];
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
            for (int j = 0; j < vectorSize && i + j < numFrames; ++j) {
                output[i + j] = outputVec[j];
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
        phase = phaseVec.back();
    }
};

//////////////////// Pulse class definition //////////////////////

template <typename sample_type> class PulseOsc : public Module<sample_type> {
  public:
    PulseOsc() : m_pulseWidth(0.5) {}

    void process(const std::vector<std::optional<sample_type *>> &inputs,
                 std::vector<sample_type *> &outputs, unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0) {
            return;
        }

        sample_type *output = outputs[0];
        std::optional<sample_type *> freqMod = !inputs.empty() ? inputs[0] : std::nullopt;
        std::optional<sample_type *> ampMod =
            inputs.size() > 1 ? inputs[1] : std::nullopt;
        std::optional<sample_type *> pwMod = inputs.size() > 2 ? inputs[2] : std::nullopt;

        sample_type phase = this->getPhase();
        sample_type baseFrequency = this->getFrequency();
        sample_type baseAmplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(AudioEngine::getSampleRate());

        processSIMD(output, freqMod, ampMod, pwMod, phase, baseFrequency, baseAmplitude,
                    sampleRate, numFrames);

        this->setPhase(phase);
    }

    [[nodiscard]] std::string getName() const override { return "Pulse Oscillator"; }

    [[nodiscard]] std::string getDescription() const override {
        return "An anti-aliased pulse wave oscillator with pulse width control";
    }

    [[nodiscard]] std::unique_ptr<Module<sample_type>> clone() const override {
        return std::make_unique<PulseOsc>(*this);
    }

    sample_type getCurrentValue() const override {
        sample_type t = this->getPhase() / Constants<sample_type>::twoPiConstant;
        sample_type dt =
            this->getFrequency() / static_cast<sample_type>(AudioEngine::getSampleRate());
        sample_type value = (t < m_pulseWidth) ? 1.0 : -1.0;
        value -= this->polyBlep(t, dt);
        value += this->polyBlep(std::fmod(t + 1.0 - m_pulseWidth, 1.0), dt);
        return this->getAmplitude() * value;
    }

    void setPulseWidth(sample_type pulseWidth) {
        m_pulseWidth = std::clamp(pulseWidth, 0.0, 1.0);
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

    void processSIMD(sample_type *output, const std::optional<sample_type *> &freqMod,
                     const std::optional<sample_type *> &ampMod,
                     const std::optional<sample_type *> &pwMod, sample_type &phase,
                     sample_type baseFrequency, sample_type baseAmplitude,
                     sample_type sampleRate, unsigned int numFrames) {
        constexpr int vectorSize = 8;
        const vec_t twoPi(vectorSize, Constants<sample_type>::twoPiConstant);
        const vec_t sampleRateVec(vectorSize, sampleRate);
        const vec_t one(vectorSize, 1.0);

        vec_t phaseVec(vectorSize, phase);
        vec_t freqVec(vectorSize, baseFrequency);
        vec_t ampVec(vectorSize, baseAmplitude);
        vec_t pwVec(vectorSize, m_pulseWidth);

        for (unsigned int i = 0; i < numFrames; i += vectorSize) {
            // Apply modulation
            if (freqMod && *freqMod) {
                for (int j = 0; j < vectorSize; ++j) {
                    freqVec[j] += (*freqMod)[i + j];
                }
            }
            if (ampMod && *ampMod) {
                for (int j = 0; j < vectorSize; ++j) {
                    ampVec[j] *= (*ampMod)[i + j];
                }
            }
            if (pwMod && *pwMod) {
                for (int j = 0; j < vectorSize; ++j) {
                    pwVec[j] = std::clamp(m_pulseWidth + (*pwMod)[i + j], 0.0, 1.0);
                }
            }

            // Calculate phase increment and normalize phase
            vec_t phaseIncVec = freqVec * twoPi / sampleRateVec;
            vec_t tVec = phaseVec / twoPi;
            vec_t dtVec = freqVec / sampleRateVec;

            // Calculate pulse wave with anti-aliasing
            vec_t pulseWave(vectorSize);
            for (int j = 0; j < vectorSize; ++j) {
                sample_type t = tVec[j];
                sample_type dt = dtVec[j];
                sample_type pw = pwVec[j];

                sample_type value = (t < pw) ? 1.0 : -1.0;
                value -= this->polyBlep(t, dt);
                value += this->polyBlep(std::fmod(t + 1.0 - pw, 1.0), dt);
                pulseWave[j] = value;
            }

            // Apply amplitude
            vec_t outputVec = ampVec * pulseWave;

            // Store result
            for (int j = 0; j < vectorSize && i + j < numFrames; ++j) {
                output[i + j] = outputVec[j];
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
        phase = phaseVec.back();
    }

    sample_type m_pulseWidth;
};

} // namespace tinysynth

#endif // OSCILLATOR_LLVM_H
