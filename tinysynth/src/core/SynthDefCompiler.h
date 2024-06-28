// SynthDefCompiler.h
#pragma once

#include "SynthDef.h"
#include "LLVMUGenBuilder.h"
#include <llvm/IR/Module.h>
#include <memory>

namespace tinysynth {

class SynthDefCompiler {
public:
    SynthDefCompiler();
    
    std::unique_ptr<llvm::Module> compile(const SynthDef& synthDef);

private:
    std::unique_ptr<llvm::LLVMContext> m_context;
    std::unique_ptr<llvm::IRBuilder<>> m_builder;
    LLVMUGenBuilder m_ugenBuilder;

    llvm::Function* createMainProcessFunction(llvm::Module* module, const SynthDef& synthDef);
    void compileUGen(const UGenInstance& ugen, llvm::Module* module);
    void connectUGens(const std::vector<Connection>& connections, llvm::Module* module);
};

} // namespace tinysynth