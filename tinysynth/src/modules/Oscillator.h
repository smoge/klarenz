#ifndef OSCILLATOR_H
#define OSCILLATOR_H

#include "../core/Module.h"
#include "../utils/Constants.h"
#include "../utils/Utils.h"
#include "AudioEngine.h"
#include <cmath>
#include <immintrin.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <optional>

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

  private:
    sample_type m_frequency{440.0};
    sample_type m_amplitude{1.0};
    sample_type m_phase{0.0};
};

template <typename sample_type> class SineOsc : public Oscillator<sample_type> {
  public:
    void process(const std::vector<sample_type *> &inputs,
                 std::vector<sample_type *> &outputs, unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0) {
            return;
        }

        sample_type *output = outputs[0];
        sample_type *freqMod = (!inputs.empty()) ? inputs[0] : nullptr;
        sample_type *ampMod = (inputs.size() > 1) ? inputs[1] : nullptr;

        sample_type phase = this->getPhase();
        sample_type baseFrequency = this->getFrequency();
        sample_type baseAmplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(AudioEngine::getSampleRate());
        const sample_type phaseIncrement =
            baseFrequency * Constants<sample_type>::twoPiConstant / sampleRate;

#if defined(__AVX__)
        // AVX optimization
        const __m256 twoPi = _mm256_set1_ps(Constants<sample_type>::twoPiConstant);
        const __m256 sampleRateVec = _mm256_set1_ps(sampleRate);
        unsigned int i = 0;
        for (; i < numFrames - 7; i += 8) {
            __m256 freqVec = _mm256_loadu_ps(&baseFrequency);
            __m256 phaseVec = _mm256_loadu_ps(&phase);
            if (freqMod) {
                freqVec = _mm256_add_ps(freqVec, _mm256_loadu_ps(&freqMod[i]));
            }
            __m256 phaseIncVec =
                _mm256_div_ps(_mm256_mul_ps(freqVec, twoPi), sampleRateVec);
            __m256 ampVec = _mm256_loadu_ps(&baseAmplitude);
            if (ampMod) {
                ampVec = _mm256_mul_ps(ampVec, _mm256_loadu_ps(&ampMod[i]));
            }
            __m256 sinVec = _mm256_sin_ps(phaseVec);
            __m256 outVec = _mm256_mul_ps(ampVec, sinVec);
            _mm256_storeu_ps(&output[i], outVec);
            phaseVec = _mm256_add_ps(phaseVec, phaseIncVec);
            for (int j = 0; j < 8; ++j) {
                if (phaseVec[j] >= Constants<sample_type>::twoPiConstant) {
                    phaseVec[j] -= Constants<sample_type>::twoPiConstant;
                }
            }
            _mm256_storeu_ps(&phase, phaseVec);
        }
        // Handle remaining samples
        for (; i < numFrames; ++i) {
            sample_type currentFrequency = baseFrequency;
            sample_type currentAmplitude = baseAmplitude;
            if (freqMod) {
                currentFrequency += freqMod[i];
            }
            if (ampMod) {
                currentAmplitude *= ampMod[i];
            }
            output[i] = currentAmplitude * std::sin(phase);
            phase +=
                currentFrequency * Constants<sample_type>::twoPiConstant / sampleRate;
            if (phase >= Constants<sample_type>::twoPiConstant) {
                phase -= Constants<sample_type>::twoPiConstant;
            }
        }
#elif defined(__SSE__)
        // SSE optimization
        const __m128 twoPi = _mm_set1_ps(Constants<sample_type>::twoPiConstant);
        const __m128 sampleRateVec = _mm_set1_ps(sampleRate);
        unsigned int i = 0;
        for (; i < numFrames - 3; i += 4) {
            __m128 freqVec = _mm_loadu_ps(&baseFrequency);
            __m128 phaseVec = _mm_loadu_ps(&phase);
            if (freqMod) {
                freqVec = _mm_add_ps(freqVec, _mm_loadu_ps(&freqMod[i]));
            }
            __m128 phaseIncVec = _mm_div_ps(_mm_mul_ps(freqVec, twoPi), sampleRateVec);
            __m128 ampVec = _mm_loadu_ps(&baseAmplitude);
            if (ampMod) {
                ampVec = _mm_mul_ps(ampVec, _mm_loadu_ps(&ampMod[i]));
            }
            __m128 sinVec = _mm_sin_ps(phaseVec);
            __m128 outVec = _mm_mul_ps(ampVec, sinVec);
            _mm_storeu_ps(&output[i], outVec);
            phaseVec = _mm_add_ps(phaseVec, phaseIncVec);
            for (int j = 0; j < 4; ++j) {
                if (phaseVec[j] >= Constants<sample_type>::twoPiConstant) {
                    phaseVec[j] -= Constants<sample_type>::twoPiConstant;
                }
            }
            _mm_storeu_ps(&phase, phaseVec);
        }
        // Handle remaining samples
        for (; i < numFrames; ++i) {
            sample_type currentFrequency = baseFrequency;
            sample_type currentAmplitude = baseAmplitude;
            if (freqMod) {
                currentFrequency += freqMod[i];
            }
            if (ampMod) {
                currentAmplitude *= ampMod[i];
            }
            output[i] = currentAmplitude * std::sin(phase);
            phase +=
                currentFrequency * Constants<sample_type>::twoPiConstant / sampleRate;
            if (phase >= Constants<sample_type>::twoPiConstant) {
                phase -= Constants<sample_type>::twoPiConstant;
            }
        }
#else
        // Fallback to scalar processing
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
            phase +=
                currentFrequency * Constants<sample_type>::twoPiConstant / sampleRate;
            if (phase >= Constants<sample_type>::twoPiConstant) {
                phase -= Constants<sample_type>::twoPiConstant;
            }
        }
#endif

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
};

/////// SawOsc class definition ///////

template <typename sample_type> class SawOsc : public Oscillator<sample_type> {
  public:
    void process(const std::vector<sample_type *> &inputs,
                 std::vector<sample_type *> &outputs, unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0) {
            return;
        }

        sample_type *output = outputs[0];
        sample_type *freqMod = (!inputs.empty()) ? inputs[0] : nullptr;
        sample_type *ampMod = (inputs.size() > 1) ? inputs[1] : nullptr;

        sample_type phase = this->getPhase();
        sample_type baseFrequency = this->getFrequency();
        sample_type baseAmplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(AudioEngine::getSampleRate());
        const sample_type phaseIncrement =
            baseFrequency * Constants<sample_type>::twoPiConstant / sampleRate;

#if defined(__AVX__)
        // AVX optimization
        const __m256 twoPi = _mm256_set1_ps(Constants<sample_type>::twoPiConstant);
        const __m256 pi = _mm256_set1_ps(Constants<sample_type>::piConstant);
        const __m256 sampleRateVec = _mm256_set1_ps(sampleRate);
        unsigned int i = 0;
        for (; i < numFrames - 7; i += 8) {
            __m256 freqVec = _mm256_set1_ps(baseFrequency);
            __m256 phaseVec = _mm256_set1_ps(phase);
            if (freqMod) {
                freqVec = _mm256_add_ps(freqVec, _mm256_loadu_ps(&freqMod[i]));
            }
            __m256 phaseIncVec =
                _mm256_div_ps(_mm256_mul_ps(freqVec, twoPi), sampleRateVec);
            __m256 ampVec = _mm256_set1_ps(baseAmplitude);
            if (ampMod) {
                ampVec = _mm256_mul_ps(ampVec, _mm256_loadu_ps(&ampMod[i]));
            }
            __m256 outVec = _mm256_mul_ps(
                ampVec, _mm256_sub_ps(_mm256_div_ps(phaseVec, pi), _mm256_set1_ps(1.0f)));
            _mm256_storeu_ps(&output[i], outVec);
            phaseVec = _mm256_add_ps(phaseVec, phaseIncVec);
            for (int j = 0; j < 8; ++j) {
                if (phaseVec[j] >= Constants<sample_type>::twoPiConstant) {
                    phaseVec[j] -= Constants<sample_type>::twoPiConstant;
                }
            }
            phase = phaseVec[7];
        }
        // Handle remaining samples
        for (; i < numFrames; ++i) {
            sample_type currentFrequency = baseFrequency;
            sample_type currentAmplitude = baseAmplitude;

            if (freqMod) {
                currentFrequency += freqMod[i];
            }
            if (ampMod) {
                currentAmplitude *= ampMod[i];
            }

            output[i] =
                currentAmplitude * ((phase / Constants<sample_type>::piConstant) - 1);
            phase +=
                currentFrequency * Constants<sample_type>::twoPiConstant / sampleRate;
            if (phase >= Constants<sample_type>::twoPiConstant) {
                phase -= Constants<sample_type>::twoPiConstant;
            }
        }
#elif defined(__SSE__)
        // SSE optimization
        const __m128 twoPi = _mm_set1_ps(Constants<sample_type>::twoPiConstant);
        const __m128 pi = _mm_set1_ps(Constants<sample_type>::piConstant);
        const __m128 sampleRateVec = _mm_set1_ps(sampleRate);
        unsigned int i = 0;
        for (; i < numFrames - 3; i += 4) {
            __m128 freqVec = _mm_set1_ps(baseFrequency);
            __m128 phaseVec = _mm_set1_ps(phase);
            if (freqMod) {
                freqVec = _mm_add_ps(freqVec, _mm_loadu_ps(&freqMod[i]));
            }
            __m128 phaseIncVec = _mm_div_ps(_mm_mul_ps(freqVec, twoPi), sampleRateVec);
            __m128 ampVec = _mm_set1_ps(baseAmplitude);
            if (ampMod) {
                ampVec = _mm_mul_ps(ampVec, _mm_loadu_ps(&ampMod[i]));
            }
            __m128 outVec = _mm_mul_ps(
                ampVec, _mm_sub_ps(_mm_div_ps(phaseVec, pi), _mm_set1_ps(1.0f)));
            _mm_storeu_ps(&output[i], outVec);
            phaseVec = _mm_add_ps(phaseVec, phaseIncVec);
            for (int j = 0; j < 4; ++j) {
                if (phaseVec[j] >= Constants<sample_type>::twoPiConstant) {
                    phaseVec[j] -= Constants<sample_type>::twoPiConstant;
                }
            }
            phase = phaseVec[3];
        }
        // Handle remaining samples
        for (; i < numFrames; ++i) {
            sample_type currentFrequency = baseFrequency;
            sample_type currentAmplitude = baseAmplitude;

            if (freqMod) {
                currentFrequency += freqMod[i];
            }
            if (ampMod) {
                currentAmplitude *= ampMod[i];
            }

            output[i] =
                currentAmplitude * ((phase / Constants<sample_type>::piConstant) - 1);
            phase +=
                currentFrequency * Constants<sample_type>::twoPiConstant / sampleRate;
            if (phase >= Constants<sample_type>::twoPiConstant) {
                phase -= Constants<sample_type>::twoPiConstant;
            }
        }
#else
        // Fallback to scalar processing
        for (unsigned int i = 0; i < numFrames; ++i) {
            sample_type currentFrequency = baseFrequency;
            sample_type currentAmplitude = baseAmplitude;

            if (freqMod) {
                currentFrequency += freqMod[i];
            }
            if (ampMod) {
                currentAmplitude *= ampMod[i];
            }

            output[i] =
                currentAmplitude * ((phase / Constants<sample_type>::piConstant) - 1);
            phase +=
                currentFrequency * Constants<sample_type>::twoPiConstant / sampleRate;
            if (phase >= Constants<sample_type>::twoPiConstant) {
                phase -= Constants<sample_type>::twoPiConstant;
            }
        }
#endif

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
        return this->getAmplitude() *
               ((this->getPhase() / Constants<sample_type>::piConstant) - 1);
    }
};

////// TriangleOsc class definition //////
template <typename sample_type> class TriangleOsc : public Oscillator<sample_type> {
  public:
    void process(const std::vector<sample_type *> &inputs,
                 std::vector<sample_type *> &outputs, unsigned int numFrames) override {
        if (outputs.empty() || numFrames == 0) {
            return;
        }

        sample_type *output = outputs[0];
        sample_type *freqMod = (!inputs.empty()) ? inputs[0] : nullptr;
        sample_type *ampMod = (inputs.size() > 1) ? inputs[1] : nullptr;

        sample_type phase = this->getPhase();
        sample_type baseFrequency = this->getFrequency();
        sample_type baseAmplitude = this->getAmplitude();
        const auto sampleRate = static_cast<sample_type>(AudioEngine::getSampleRate());
        const sample_type phaseIncrement =
            baseFrequency * Constants<sample_type>::twoPiConstant / sampleRate;

#if defined(__AVX__)
        // AVX optimization
        const __m256 twoPi = _mm256_set1_ps(Constants<sample_type>::twoPiConstant);
        const __m256 pi = _mm256_set1_ps(Constants<sample_type>::piConstant);
        const __m256 sampleRateVec = _mm256_set1_ps(sampleRate);
        const __m256 two = _mm256_set1_ps(2.0f);
        const __m256 three = _mm256_set1_ps(3.0f);
        unsigned int i = 0;
        for (; i < numFrames - 7; i += 8) {
            __m256 freqVec = _mm256_set1_ps(baseFrequency);
            __m256 phaseVec = _mm256_set1_ps(phase);
            if (freqMod) {
                freqVec = _mm256_add_ps(freqVec, _mm256_loadu_ps(&freqMod[i]));
            }
            __m256 phaseIncVec =
                _mm256_div_ps(_mm256_mul_ps(freqVec, twoPi), sampleRateVec);
            __m256 ampVec = _mm256_set1_ps(baseAmplitude);
            if (ampMod) {
                ampVec = _mm256_mul_ps(ampVec, _mm256_loadu_ps(&ampMod[i]));
            }
            __m256 outVec = _mm256_blendv_ps(
                _mm256_sub_ps(_mm256_mul_ps(two, _mm256_div_ps(phaseVec, pi)),
                              _mm256_set1_ps(1.0f)),
                _mm256_sub_ps(three, _mm256_mul_ps(two, _mm256_div_ps(phaseVec, pi))),
                _mm256_cmp_ps(phaseVec, pi, _CMP_GE_OS));
            outVec = _mm256_mul_ps(ampVec, outVec);
            _mm256_storeu_ps(&output[i], outVec);
            phaseVec = _mm256_add_ps(phaseVec, phaseIncVec);
            for (int j = 0; j < 8; ++j) {
                if (phaseVec[j] >= Constants<sample_type>::twoPiConstant) {
                    phaseVec[j] -= Constants<sample_type>::twoPiConstant;
                }
            }
            phase = phaseVec[7];
        }
        // Handle remaining samples
        for (; i < numFrames; ++i) {
            sample_type currentFrequency = baseFrequency;
            sample_type currentAmplitude = baseAmplitude;

            if (freqMod) {
                currentFrequency += freqMod[i];
            }
            if (ampMod) {
                currentAmplitude *= ampMod[i];
            }

            output[i] = currentAmplitude *
                        ((phase < Constants<sample_type>::piConstant)
                             ? -1 + (2 * phase / Constants<sample_type>::piConstant)
                             : 3 - (2 * phase / Constants<sample_type>::piConstant));
            phase +=
                currentFrequency * Constants<sample_type>::twoPiConstant / sampleRate;
            if (phase >= Constants<sample_type>::twoPiConstant) {
                phase -= Constants<sample_type>::twoPiConstant;
            }
        }
#elif defined(__SSE__)
        // SSE optimization
        const __m128 twoPi = _mm_set1_ps(Constants<sample_type>::twoPiConstant);
        const __m128 pi = _mm_set1_ps(Constants<sample_type>::piConstant);
        const __m128 sampleRateVec = _mm_set1_ps(sampleRate);
        const __m128 two = _mm_set1_ps(2.0f);
        const __m128 three = _mm_set1_ps(3.0f);
        unsigned int i = 0;
        for (; i < numFrames - 3; i += 4) {
            __m128 freqVec = _mm_set1_ps(baseFrequency);
            __m128 phaseVec = _mm_set1_ps(phase);
            if (freqMod) {
                freqVec = _mm_add_ps(freqVec, _mm_loadu_ps(&freqMod[i]));
            }
            __m128 phaseIncVec = _mm_div_ps(_mm_mul_ps(freqVec, twoPi), sampleRateVec);
            __m128 ampVec = _mm_set1_ps(baseAmplitude);
            if (ampMod) {
                ampVec = _mm_mul_ps(ampVec, _mm_loadu_ps(&ampMod[i]));
            }
            __m128 outVec = _mm_blendv_ps(
                _mm_sub_ps(_mm_mul_ps(two, _mm_div_ps(phaseVec, pi)), _mm_set1_ps(1.0f)),
                _mm_sub_ps(three, _mm_mul_ps(two, _mm_div_ps(phaseVec, pi))),
                _mm_cmpge_ps(phaseVec, pi));
            outVec = _mm_mul_ps(ampVec, outVec);
            _mm_storeu_ps(&output[i], outVec);
            phaseVec = _mm_add_ps(phaseVec, phaseIncVec);
            for (int j = 0; j < 4; ++j) {
                if (phaseVec[j] >= Constants<sample_type>::twoPiConstant) {
                    phaseVec[j] -= Constants<sample_type>::twoPiConstant;
                }
            }
            phase = phaseVec[3];
        }
        // Handle remaining samples
        for (; i < numFrames; ++i) {
            sample_type currentFrequency = baseFrequency;
            sample_type currentAmplitude = baseAmplitude;

            if (freqMod) {
                currentFrequency += freqMod[i];
            }
            if (ampMod) {
                currentAmplitude *= ampMod[i];
            }

            output[i] = currentAmplitude *
                        ((phase < Constants<sample_type>::piConstant)
                             ? -1 + (2 * phase / Constants<sample_type>::piConstant)
                             : 3 - (2 * phase / Constants<sample_type>::piConstant));
            phase +=
                currentFrequency * Constants<sample_type>::twoPiConstant / sampleRate;
            if (phase >= Constants<sample_type>::twoPiConstant) {
                phase -= Constants<sample_type>::twoPiConstant;
            }
        }
#else
        // Fallback to scalar processing
        for (unsigned int i = 0; i < numFrames; ++i) {
            sample_type currentFrequency = baseFrequency;
            sample_type currentAmplitude = baseAmplitude;

            if (freqMod) {
                currentFrequency += freqMod[i];
            }
            if (ampMod) {
                currentAmplitude *= ampMod[i];
            }

            output[i] = currentAmplitude *
                        ((phase < Constants<sample_type>::piConstant)
                             ? -1 + (2 * phase / Constants<sample_type>::piConstant)
                             : 3 - (2 * phase / Constants<sample_type>::piConstant));
            phase +=
                currentFrequency * Constants<sample_type>::twoPiConstant / sampleRate;
            if (phase >= Constants<sample_type>::twoPiConstant) {
                phase -= Constants<sample_type>::twoPiConstant;
            }
        }
#endif

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
        return this->getAmplitude() *
               ((phase < Constants<sample_type>::piConstant)
                    ? -1 + (2 * phase / Constants<sample_type>::piConstant)
                    : 3 - (2 * phase / Constants<sample_type>::piConstant));
    }
};

} // namespace tinysynth

#endif // OSCILLATOR_H
