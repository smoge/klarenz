#include <iostream> 
#include "ModularSystem.h"
#include <algorithm>
#include <optional>
#include <stdexcept>

namespace tinysynth {


void logConnection(const std::string &fromModule, unsigned int outputIndex,
                   const std::string &toModule, unsigned int inputIndex) {
    // Simple logging function --->> debugging purposes
    std::cout << "Connecting " << fromModule << "[" << outputIndex << "] -> "
              << toModule << "[" << inputIndex << "]" << '\n';
}

bool validateModuleName(const std::string &name) {

    return !name.empty();
}

} // namespace tinysynth
