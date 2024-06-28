#ifndef UTILITY_H
#define UTILITY_H

#include <algorithm>  // For std::max and std::min

namespace util {

template <typename T>
static T clamp(T value, T min, T max) {
    return std::max(min, std::min(value, max));
}

} // namespace util

#endif // UTILITY_H
