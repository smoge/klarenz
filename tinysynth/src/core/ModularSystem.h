// ModularSystem.h
#ifndef MODULARSYSTEM_H
#define MODULARSYSTEM_H

#include "Module.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinysynth {

template <typename sample_type> class ModularSystem {
  public:
    ModularSystem() = default;
    ModularSystem(const ModularSystem &) = delete;
    ModularSystem(ModularSystem &&) = delete;
    ModularSystem &operator=(const ModularSystem &) = delete;
    ModularSystem &operator=(ModularSystem &&) = delete;
    ~ModularSystem() = default;

    // Add a module to the system
    void addModule(const std::string &name, std::unique_ptr<Module<sample_type>> module);

    // Remove a module from the system
    void removeModule(const std::string &name);

    // Connect two modules
    void connect(const std::string &fromModule, unsigned int outputIndex,
                 const std::string &toModule, unsigned int inputIndex);

    // Disconnect two modules
    void disconnect(const std::string &fromModule, unsigned int outputIndex,
                    const std::string &toModule, unsigned int inputIndex);

    // Process audio through the entire system
    void process(unsigned int numFrames);

    // Get a list of all module names
    [[nodiscard]] std::vector<std::string> getModuleNames() const;

    // Get a pointer to a specific module
    Module<sample_type> *getModule(const std::string &name);

  private:
    struct Connection {
        std::string fromModule;
        unsigned int outputIndex;
        std::string toModule;
        unsigned int inputIndex;
    };

    std::unordered_map<std::string, std::unique_ptr<Module<sample_type>>> m_modules;
    std::vector<Connection> m_connections;
    std::vector<std::vector<sample_type>> m_audioBuffers;
};

} // namespace tinysynth

#endif // MODULARSYSTEM_H