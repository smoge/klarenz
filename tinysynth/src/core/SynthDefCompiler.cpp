// SynthDefCompiler.cpp
#include "SynthDefCompiler.h"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Linker/Linker.h>

namespace tinysynth {

SynthDefCompiler::SynthDefCompiler()
    : m_context(std::make_unique<llvm::LLVMContext>()),
      m_builder(std::make_unique<llvm::IRBuilder<>>(*m_context)),
      m_ugenBuilder() {}

std::unique_ptr<llvm::Module>
SynthDefCompiler::compile(const SynthDef &synthDef) {
  auto module = std::make_unique<llvm::Module>("SynthDef", *m_context);

  // Create the main process function
  llvm::Function *mainFunc = createMainProcessFunction(module.get(), synthDef);

  // Compile each UGen
  for (const auto &ugen : synthDef.getUGens()) {
    compileUGen(ugen, module.get());
  }

  // Connect UGens
  connectUGens(synthDef.getConnections(), module.get());

  // Verify the module
  std::string errorInfo;
  llvm::raw_string_ostream errorStream(errorInfo);
  if (llvm::verifyModule(*module, &errorStream)) {
    throw std::runtime_error("Module verification failed: " + errorInfo);
  }


  // Optimize the module
  llvm::PassManagerBuilder PMBuilder;
  PMBuilder.OptLevel = 3;
  llvm::legacy::FunctionPassManager FPM(module.get());
  PMBuilder.populateFunctionPassManager(FPM);

  FPM.doInitialization();

  // If you want to run module-level passes as well:
  llvm::legacy::PassManager MPM;
  PMBuilder.populateModulePassManager(MPM);
  MPM.run(*module);

  for (auto &F : *module) {
    FPM.run(F);
  }

  return module;
}

llvm::Function *
SynthDefCompiler::createMainProcessFunction(llvm::Module *module,
                                            const SynthDef &synthDef) {
  // Create function signature: void process(float* input, float* output, int
  // numFrames)
  llvm::FunctionType *funcType =
      llvm::FunctionType::get(llvm::Type::getVoidTy(*m_context),
                              {llvm::Type::getFloatPtrTy(*m_context),
                               llvm::Type::getFloatPtrTy(*m_context),
                               llvm::Type::getInt32Ty(*m_context)},
                              false);

  llvm::Function *func = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, "process", module);

  // Create the entry basic block
  llvm::BasicBlock *entryBlock =
      llvm::BasicBlock::Create(*m_context, "entry", func);
  m_builder->SetInsertPoint(entryBlock);

  // TODO: Implement the main processing loop here

  m_builder->CreateRetVoid();
  return func;
}

void SynthDefCompiler::compileUGen(const UGenInstance &ugen,
                                   llvm::Module *module) {
  std::unique_ptr<llvm::Module> ugenModule;

  if (ugen.ugenType == "SineOsc") {
    ugenModule = m_ugenBuilder.buildSineOsc();
  } else if (ugen.ugenType == "SawOsc") {
    ugenModule = m_ugenBuilder.buildSawOsc();
  } else if (ugen.ugenType == "TriangleOsc") {
    ugenModule = m_ugenBuilder.buildTriangleOsc();
  } else if (ugen.ugenType == "PulseOsc") {
    ugenModule = m_ugenBuilder.buildPulseOsc();
  } else {
    throw std::runtime_error("Unknown UGen type: " + ugen.ugenType);
  }

  // Verify the UGen module
  std::string errorInfo;
  llvm::raw_string_ostream errorStream(errorInfo);
  if (llvm::verifyModule(*ugenModule, &errorStream)) {
    throw std::runtime_error("UGen module verification failed: " + errorInfo);
  }

  // Link the UGen module into the main module
  bool linkError = llvm::Linker::linkModules(*module, std::move(ugenModule));
  if (linkError) {
    throw std::runtime_error("Failed to link UGen module for " + ugen.ugenType);
  }

  // Find the linked UGen function
  llvm::Function *ugenFunc = module->getFunction(ugen.ugenType + "_process");
  if (!ugenFunc) {
    throw std::runtime_error("Failed to find linked UGen function for " +
                             ugen.ugenType);
  }

  // Set UGen parameters
  for (const auto &[paramName, paramValue] : ugen.parameters) {
    llvm::Function *setParamFunc =
        module->getFunction(ugen.ugenType + "_setParameter");
    if (setParamFunc) {
      std::vector<llvm::Value *> args = {
          llvm::ConstantFP::get(m_builder->getFloatTy(), paramValue)};
      m_builder->CreateCall(setParamFunc, args);
    }
  }
}

void SynthDefCompiler::connectUGens(const std::vector<Connection> &connections,
                                    llvm::Module *module) {
  // TODO: Implement UGen connections
  // This will involve creating function calls between UGens in the LLVM IR
}

} // namespace tinysynth