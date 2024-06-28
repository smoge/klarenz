// ModularSystem.h
#ifndef MODULARSYSTEM_H
#define MODULARSYSTEM_H

#include "Module.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <optional>

namespace tinysynth {

template <typename sample_type>
class ModularSystem {
public:
    ModularSystem() = default;
    ModularSystem(const ModularSystem &) = delete;
    ModularSystem(ModularSystem &&) = delete;
    ModularSystem &operator=(const ModularSystem &) = delete;
    ModularSystem &operator=(ModularSystem &&) = delete;
    ~ModularSystem() = default;

    // Add a module to the system
    void addModule(const std::string& name, std::unique_ptr<Module<sample_type>> module);

    // Remove a module from the system
    void removeModule(const std::string& name);

    // Connect two modules
    void connect(const std::string& fromModule, unsigned int outputIndex,
                 const std::string& toModule, unsigned int inputIndex);

    // Disconnect two modules
    void disconnect(const std::string& fromModule, unsigned int outputIndex,
                    const std::string& toModule, unsigned int inputIndex);

    // Process audio through the entire system
    void process(unsigned int numFrames);

    // Get a list of all module names
    [[nodiscard]] std::vector<std::string> getModuleNames() const;

    // Get a pointer to a specific module
    Module<sample_type>* getModule(const std::string& name);

private:
    struct Connection {
        std::string fromModule;
        unsigned int outputIndex{};
        std::string toModule;
        unsigned int inputIndex{};
    };

    std::unordered_map<std::string, std::unique_ptr<Module<sample_type>>> m_modules;
    std::vector<Connection> m_connections;
    std::vector<std::vector<sample_type>> m_audioBuffers;
};

template <typename sample_type>
void ModularSystem<sample_type>::process(unsigned int numFrames) {
    // Resize audio buffers if necessary
    m_audioBuffers.resize(m_modules.size());
    for (auto& buffer : m_audioBuffers) {
        buffer.resize(numFrames);
    }

    // Process each module
    for (const auto& [name, module] : m_modules) {
        std::vector<std::optional<sample_type*>> inputs;
        std::vector<sample_type*> outputs;

        // Prepare inputs
        inputs.resize(module->getNumInputs(), std::nullopt);

        // Set up connections
        for (const auto& conn : m_connections) {
            if (conn.toModule == name) {
                auto fromModuleIt = m_modules.find(conn.fromModule);
                if (fromModuleIt != m_modules.end()) {
                    auto bufferIndex = std::distance(m_modules.begin(), fromModuleIt);
                    inputs[conn.inputIndex] = m_audioBuffers[bufferIndex].data();
                }
            }
        }

        // Prepare outputs
        auto moduleIndex = std::distance(m_modules.begin(), m_modules.find(name));
        for (unsigned int i = 0; i < module->getNumOutputs(); ++i) {
            outputs.push_back(m_audioBuffers[moduleIndex].data());
        }

        // Process the module
        module->process(inputs, outputs, numFrames);
    }
}

} // namespace tinysynth

#endif // MODULARSYSTEM_H