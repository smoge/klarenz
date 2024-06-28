
namespace tinysynth {

class AudioEngine {
public:
    static void setSampleRate(unsigned int sampleRate) {
        s_sampleRate = sampleRate;
    }

    static unsigned int getSampleRate() {
        return s_sampleRate;
    }

private:
    static unsigned int s_sampleRate;
};

} // namespace tinysynth

