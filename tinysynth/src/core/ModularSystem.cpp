// ModularSystem.cpp
#include "ModularSystem.h"
#include <algorithm>
#include <stdexcept>

namespace tinysynth {

template <typename sample_type>
void ModularSystem<sample_type>::addModule(const std::string &name,
                                           std::unique_ptr<Module<sample_type>> module) {
    if (m_modules.find(name) != m_modules.end()) {
        throw std::runtime_error("Module with name '" + name + "' already exists.");
    }
    m_modules[name] = std::move(module);
}

template <typename sample_type>
void ModularSystem<sample_type>::removeModule(const std::string &name) {
    auto it = m_modules.find(name);
    if (it == m_modules.end()) {
        throw std::runtime_error("Module with name '" + name + "' does not exist.");
    }

    // Remove all connections involving this module
    m_connections.erase(std::remove_if(m_connections.begin(), m_connections.end(),
                                       [&name](const Connection &conn) {
                                           return conn.fromModule == name ||
                                                  conn.toModule == name;
                                       }),
                        m_connections.end());

    m_modules.erase(it);
}

template <typename sample_type>
void ModularSystem<sample_type>::connect(const std::string &fromModule,
                                         unsigned int outputIndex,
                                         const std::string &toModule,
                                         unsigned int inputIndex) {
    if (m_modules.find(fromModule) == m_modules.end()) {
        throw std::runtime_error("Module '" + fromModule + "' does not exist.");
    }
    if (m_modules.find(toModule) == m_modules.end()) {
        throw std::runtime_error("Module '" + toModule + "' does not exist.");
    }

    m_connections.push_back({fromModule, outputIndex, toModule, inputIndex});
}

template <typename sample_type>
void ModularSystem<sample_type>::disconnect(const std::string &fromModule,
                                            unsigned int outputIndex,
                                            const std::string &toModule,
                                            unsigned int inputIndex) {
    auto it = std::find_if(
        m_connections.begin(), m_connections.end(), [&](const Connection &conn) {
            return conn.fromModule == fromModule && conn.outputIndex == outputIndex &&
                   conn.toModule == toModule && conn.inputIndex == inputIndex;
        });

    if (it != m_connections.end()) {
        m_connections.erase(it);
    }
}

template <typename sample_type>
void ModularSystem<sample_type>::process(unsigned int numFrames) {
    // Resize audio buffers if necessary
    m_audioBuffers.resize(m_modules.size());
    for (auto &buffer : m_audioBuffers) {
        buffer.resize(numFrames);
    }

    // Process each module
    for (const auto &[name, module] : m_modules) {
        std::vector<std::optional<sample_type *>> inputs;
        std::vector<sample_type *> outputs;

        // Prepare inputs
        for (unsigned int i = 0; i < module->getNumInputs(); ++i) {
            inputs.push_back(nullptr); // Initialize with no connection
        }

        // Set up connections
        for (const auto &conn : m_connections) {
            if (conn.toModule == name) {
                auto fromModuleIt = m_modules.find(conn.fromModule);
                if (fromModuleIt != m_modules.end()) {
                    inputs[conn.inputIndex] =
                        m_audioBuffers[std::distance(m_modules.begin(), fromModuleIt)]
                            .data();
                }
            }
        }

        // Prepare outputs
        for (unsigned int i = 0; i < module->getNumOutputs(); ++i) {
            outputs.push_back(
                m_audioBuffers[std::distance(m_modules.begin(), m_modules.find(name))]
                    .data());
        }

        // Process the module
        module->process(inputs, outputs, numFrames);
    }
}

template <typename sample_type>
std::vector<std::string> ModularSystem<sample_type>::getModuleNames() const {
    std::vector<std::string> names;
    names.reserve(m_modules.size());
    for (const auto &[name, _] : m_modules) {
        names.push_back(name);
    }
    return names;
}

template <typename sample_type>
Module<sample_type> *ModularSystem<sample_type>::getModule(const std::string &name) {
    auto it = m_modules.find(name);
    if (it == m_modules.end()) {
        return nullptr;
    }
    return it->second.get();
}

// Explicit template instantiation for the types we'll use
template class ModularSystem<float>;
template class ModularSystem<double>;

} // namespace tinysynth