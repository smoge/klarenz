#ifndef PHASOR_H
#define PHASOR_H

#include <type_traits>
#include <cmath>

namespace tinysynth {
namespace detail {

/** Template version of phasor compute sample */
template<typename sample_type, typename internal_type>
inline sample_type phasorComputeSample(internal_type & phase, const internal_type & phase_inc, const internal_type wrap) {
    phase += phase_inc;
    phase = std::fmod(phase, wrap);
    return static_cast<sample_type>(phase);
}

/** Template metaprogramming version of the phasor loop */
template<unsigned int n, typename sample_type, typename internal_type>
constexpr void phasorPerform(const sample_type * in, sample_type * out, internal_type & phase,
                              const internal_type freq_factor, const internal_type wrap) {
    if constexpr (n > 0) {
        const internal_type phase_inc = internal_type(*in) * freq_factor;
        *out = phasorComputeSample<sample_type, internal_type>(phase, phase_inc, wrap);
        phasorPerform<n-1>(in+1, out+1, phase, freq_factor, wrap);
    }
}

template<unsigned int n, typename sample_type, typename internal_type>
constexpr void phasorPerform(sample_type * out, internal_type & phase, const internal_type phase_inc,
                              const internal_type wrap) {
    if constexpr (n > 0) {
        *out = phasorComputeSample<sample_type, internal_type>(phase, phase_inc, wrap);
        phasorPerform<n-1>(out+1, phase, phase_inc, wrap);
    }
}

} // namespace detail

template <typename sample_type = float,
          typename internal_type = sample_type,
          unsigned int wrap_arg = 1>
class Phasor {
    static_assert(std::is_floating_point_v<sample_type>, "sample_type must be floating point");
    static_assert(std::is_floating_point_v<internal_type>, "internal_type must be floating point");

public:
    Phasor() : phase_(0), phase_inc_(0), freq_factor(0) {}

    template <typename input_buffer_type, typename output_buffer_type>
    inline void perform(input_buffer_type const & in, output_buffer_type & out, unsigned int n) {
        const sample_type *inSample = in.data();
        sample_type *outSample = out.data();

        internal_type phase = phase_;
        do {
            internal_type phase_inc = internal_type(*inSample) * freq_factor;

            *outSample = computeSample(phase, phase_inc);

            ++outSample;
            ++inSample;
            --n;
        } while (n);
        phase_ = phase;
    }

    template <typename input_buffer_type, typename output_buffer_type>
    inline void perform8(input_buffer_type const & in, output_buffer_type & out, unsigned int n) {
        const sample_type *inSample = in.data();
        sample_type *outSample = out.data();

        internal_type phase = phase_;
        n /= 8;
        do {
            detail::phasorPerform<8>(inSample, outSample, phase, freq_factor, wrap());

            outSample += 8;
            inSample += 8;
            --n;
        } while (n);
        phase_ = phase;
    }

    template <unsigned int n, typename input_buffer_type, typename output_buffer_type>
    inline void perform(input_buffer_type const & in, output_buffer_type & out) {
        const sample_type *inSample = in.data();
        sample_type *outSample = out.data();

        internal_type phase = phase_;
        detail::phasorPerform<n>(inSample, outSample, phase, freq_factor, wrap());
        phase_ = phase;
    }

    template <typename output_buffer_type>
    inline void perform(output_buffer_type & out, unsigned int n) {
        sample_type *outSample = out.data();

        internal_type phase = phase_;
        do {
            *outSample = computeSample(phase, phase_inc_);

            ++outSample;
            --n;
        } while (n);
        phase_ = phase;
    }

    template <typename output_buffer_type>
    inline void perform8(output_buffer_type & out, unsigned int n) {
        sample_type *outSample = out.data();

        internal_type phase = phase_;
        n /= 8;
        do {
            detail::phasorPerform<8>(outSample, phase, phase_inc_, wrap());

            outSample += 8;
            --n;
        } while (n);
        phase_ = phase;
    }

    template <unsigned int n, typename output_buffer_type>
    inline void perform(output_buffer_type & out) {
        sample_type *outSample = out.data();

        internal_type phase = phase_;
        detail::phasorPerform<n>(outSample, phase, phase_inc_, wrap());

        phase_ = phase;
    }

    void setFrequency(internal_type const & new_frequency) {
        phase_inc_ = new_frequency * wrap();
    }

    void setPhase(internal_type const & new_phase) {
        phase_ = new_phase;
    }

    void setInvSamplerate(internal_type const & inv_samplerate) {
        freq_factor = inv_samplerate * wrap();
    }

protected:
    inline static internal_type wrap() {
        return internal_type(wrap_arg);
    }

    static inline sample_type computeSample(internal_type & phase, const internal_type phase_inc) {
        return detail::phasorComputeSample<sample_type, internal_type>(phase, phase_inc, wrap());
    }

protected:
    internal_type phase_;
    internal_type phase_inc_;
    internal_type freq_factor;
};

} // namespace tinysynth

#endif // PHASOR_H
