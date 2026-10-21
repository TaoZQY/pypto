/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "pypto/hierarchy/level_handler.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/ir/function.h"

namespace pypto {
namespace hierarchy {
namespace {

class HostLevelHandler final : public LevelHandler {
 public:
  std::string GetDomainAllocator() const override { return "(_domain_provider or orch.allocate_domain)"; }

  std::string FormatDomainMembership(const std::vector<int64_t>& children) const override {
    std::ostringstream workers;
    workers << "[";
    if (children.empty()) {
      workers << "*range(world_size)";
    } else {
      for (size_t i = 0; i < children.size(); ++i) {
        if (i > 0) workers << ", ";
        workers << children[i];
      }
    }
    workers << "]";
    return workers.str();
  }

  std::string FormatCommCtxValue(const std::string& table, const std::string& child) const override {
    return table + "[" + child + "].device_ctx";
  }

  std::string FormatHostTensorAppend(const HostArgSpec& spec) const override {
    return "add_chip_tensor_arg(" + spec.task_args + ", " + spec.worker_expr + ", " + spec.tensor_expr +
           ", " + spec.tag + ", callables, \"" + spec.callee_name + "\", " +
           std::to_string(spec.tensor_index) + ")";
  }

  bool MaterializesWindowTensor() const override { return true; }

  std::vector<std::string> GetEntryKeywordParams(bool accepts_domain_provider) const override {
    // Keep the runner ABI and its ordering, including the persistent-domain hook.
    std::vector<std::string> params = {"tensors", "callables", "sub_ids", "_keep", "world_size"};
    if (accepts_domain_provider) params.push_back("_domain_provider=None");
    return params;
  }

  std::vector<std::string> GetImports() const override {
    // DataType builds address-free window tensors; CommBufferSpec describes
    // domain allocations. Import both even for comm-less programs, as before.
    // _submit_chip also namespaces per-dispatch DFX output_prefix values.
    return {"import torch",
            "from simpler.task_interface import "
            "CallConfig, CommBufferSpec, DataType, TaskArgs, TensorArgType",
            "from pypto.runtime.tensor_arg import make_tensor_arg",
            "from pypto.runtime.distributed_runner import add_chip_tensor_arg, _submit_chip"};
  }
};

}  // namespace

const LevelHandler* GetLevelHandler(ir::Level level) {
  if (level != ir::Level::HOST) return nullptr;
  static const HostLevelHandler handler;
  return &handler;
}

}  // namespace hierarchy
}  // namespace pypto
