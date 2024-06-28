#ifndef AUDIO_ENGINE_H
#define AUDIO_ENGINE_H

namespace tinysynth {

class AudioEngine {
public:
    static void setSampleRate(unsigned int sampleRate) {
        m_sampleRate = sampleRate;
    }

    static unsigned int getSampleRate() {
        return m_sampleRate;
    }

private:
    static unsigned int m_sampleRate;
};

} // namespace tinysynth

#endif // AUDIO_ENGINE_H
