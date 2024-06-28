// LLVMUGenBuilder.h
#pragma once

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <memory>
#include <string>

namespace tinysynth {

class LLVMUGenBuilder {
public:
    LLVMUGenBuilder();

    std::unique_ptr<llvm::Module> buildSineOsc();
    std::unique_ptr<llvm::Module> buildSawOsc();
    std::unique_ptr<llvm::Module> buildTriangleOsc();
    std::unique_ptr<llvm::Module> buildPulseOsc();

private:
    std::unique_ptr<llvm::LLVMContext> m_context;
    std::unique_ptr<llvm::IRBuilder<>> m_builder;

    llvm::Function* createProcessFunction(llvm::Module* module);
    void addPhaseAccumulation(llvm::Function* func);
};

} // namespace tinysynth