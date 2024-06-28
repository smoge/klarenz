// SynthDef.h
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace tinysynth {

struct UGenInstance {
    std::string ugenType;
    std::string instanceName;
    std::unordered_map<std::string, float> parameters;
};

struct Connection {
    std::string fromUGen;
    unsigned int outputIndex;
    std::string toUGen;
    unsigned int inputIndex;
};

class SynthDef {
public:
    void addUGen(const UGenInstance& ugen);
    void addConnection(const Connection& connection);
    void setParameter(const std::string& ugenName, const std::string& paramName, float value);
    
    const std::vector<UGenInstance>& getUGens() const { return m_ugens; }
    const std::vector<Connection>& getConnections() const { return m_connections; }

private:
    std::vector<UGenInstance> m_ugens;
    std::vector<Connection> m_connections;
};

} // namespace tinysynth