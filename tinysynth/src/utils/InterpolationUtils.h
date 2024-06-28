#ifndef INTERPOLATION_UTILS_H
#define INTERPOLATION_UTILS_H

#include <vector>
#include <cmath>
#include <type_traits>

namespace tinysynth {

constexpr unsigned int TABLE_SIZE = 2048;

template<typename T>
class InterpolationUtils {
    static_assert(std::is_floating_point<T>::value, "Template parameter must be a floating-point type.");

public:
    static std::vector<T> createSineTable() {
        std::vector<T> table(TABLE_SIZE);
        for (unsigned int i = 0; i < TABLE_SIZE; ++i) {
            table[i] = static_cast<T>(std::sin(2.0 * M_PI * i / TABLE_SIZE));
        }
        return table;
    }

    static T cubicInterpolate(T v0, T v1, T v2, T v3, T t) {
        T P = (v3 - v2) - (v0 - v1);
        T Q = (v0 - v1) - P;
        T R = v2 - v0;
        T S = v1;
        return P * t * t * t + Q * t * t + R * t + S;
    }

    static T interpolate(const std::vector<T>& table, T index) {
        int idx = static_cast<int>(index);
        T frac = index - idx;
        T v0 = table[(idx - 1 + TABLE_SIZE) % TABLE_SIZE];
        T v1 = table[idx];
        T v2 = table[(idx + 1) % TABLE_SIZE];
        T v3 = table[(idx + 2) % TABLE_SIZE];
        return cubicInterpolate(v0, v1, v2, v3, frac);
    }
};

using InterpolationUtilsF = InterpolationUtils<float>;
using InterpolationUtilsD = InterpolationUtils<double>;

} // namespace tinysynth

#endif // INTERPOLATION_UTILS_H
