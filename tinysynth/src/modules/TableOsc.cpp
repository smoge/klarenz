#ifndef LOOKUP_TABLE_OSCILLATOR_H
#define LOOKUP_TABLE_OSCILLATOR_H

#include "Phasor.h"
#include <cmath>
#include <iostream>
#include <vector>

template <typename sample_type> constexpr unsigned int tableSize = 2048;

template <typename sample_type> std::vector<sample_type> createSineTable() {
    std::vector<sample_type> table(tableSize<sample_type>);
    for (unsigned int i = 0; i < tableSize<sample_type>; ++i) {
        table[i] = std::sin(2.0 * M_PI * i / tableSize<sample_type>);
    }
    return table;
}

template <typename sample_type>
sample_type cubicInterpolate(sample_type v0, sample_type v1, sample_type v2,
                             sample_type v3, sample_type t) {
    sample_type P = (v3 - v2) - (v0 - v1);
    sample_type Q = (v0 - v1) - P;
    sample_type R = v2 - v0;
    sample_type S = v1;
    return P * t * t * t + Q * t * t + R * t + S;
}

template <typename sample_type>
sample_type interpolate(const std::vector<sample_type> &table, sample_type index) {
    int idx = static_cast<int>(index);
    sample_type frac = index - idx;
    sample_type v0 = table[(idx - 1 + tableSize<sample_type>) % tableSize<sample_type>];
    sample_type v1 = table[idx];
    sample_type v2 = table[(idx + 1) % tableSize<sample_type>];
    sample_type v3 = table[(idx + 2) % tableSize<sample_type>];
    return cubicInterpolate(v0, v1, v2, v3, frac);
}

template <typename sample_type = float, typename internal_type = sample_type>
class LookupTableOscillator {
  public:
    LookupTableOscillator() : m_phasor(), m_sineTable(createSineTable<sample_type>()) {}

    void setFrequency(internal_type frequency) { m_phasor.setFrequency(frequency); }

    void setPhase(internal_type phase) { m_phasor.setPhase(phase); }

    void setInvSamplerate(internal_type invSamplerate) {
        m_phasor.setInvSamplerate(invSamplerate);
    }

    void process(const std::vector<sample_type> &input, std::vector<sample_type> &output,
                 unsigned int numFrames) {
        std::vector<internal_type> phaseBuffer(numFrames);
        m_phasor.perform(input, phaseBuffer, numFrames);

        for (unsigned int i = 0; i < numFrames; ++i) {
            sample_type phase = phaseBuffer[i] * tableSize<sample_type> / (2.0 * M_PI);
            output[i] = interpolate(m_sineTable, phase);
        }
    }

  private:
    tinysynth::Phasor<sample_type, internal_type> m_phasor;
    std::vector<sample_type> m_sineTable;
};

#endif // LOOKUP_TABLE_OSCILLATOR_H

// int main() {
//     // Example with float
//     LookupTableOscillator<float> oscFloat;
//     oscFloat.setFrequency(440.0f);
//     oscFloat.setInvSamplerate(1.0f / 48000.0f);

//     std::vector<float> inputBufferFloat(256, 1.0f); // Example input buffer with 256
//     samples std::vector<float> outputBufferFloat(256);      // Output buffer to store
//     processed samples

//     oscFloat.process(inputBufferFloat, outputBufferFloat, inputBufferFloat.size());

//     std::cout << "Output with float:\n";
//     for (float sample : outputBufferFloat) {
//         std::cout << sample << " ";
//     }
//     std::cout << std::endl;

//     // Example with double
//     LookupTableOscillator<double> oscDouble;
//     oscDouble.setFrequency(440.0);
//     oscDouble.setInvSamplerate(1.0 / 48000.0);

//     std::vector<double> inputBufferDouble(256, 1.0); // Example input buffer with 256
//     samples std::vector<double> outputBufferDouble(256);      // Output buffer to store
//     processed samples

//     oscDouble.process(inputBufferDouble, outputBufferDouble, inputBufferDouble.size());

//     std::cout << "Output with double:\n";
//     for (double sample : outputBufferDouble) {
//         std::cout << sample << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }
