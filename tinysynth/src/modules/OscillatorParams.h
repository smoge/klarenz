// OscillatorParams.h

#pragma once

template <typename sample_type>
struct OscillatorParams {
    sample_type phase;
    sample_type baseFrequency;
    sample_type baseAmplitude;
    sample_type sampleRate;
};

template <typename sample_type>
struct SIMDProcessingParams {
    sample_type *output;
    std::optional<sample_type *> freqMod;
    std::optional<sample_type *> ampMod;
    unsigned int numFrames;
};
