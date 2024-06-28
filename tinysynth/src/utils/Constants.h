#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace tinysynth {

template <typename T>
struct Constants {
    static constexpr T piConstant = static_cast<T>(3.14159265358979323846); 
    static constexpr T twoPiConstant = static_cast<T>(2.0) * piConstant; 
};

} // namespace tinysynth

#endif // CONSTANTS_H
