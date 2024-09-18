// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <iosfwd>

#include <absl/strings/str_join.h>

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace webgpu {
class ShaderHelper;
class ComputeContext;
class WebGpuContext;

// data type of uniform variable
enum class ProgramUniformVariableDataType {
  Float32,
  Float16,
  Uint32,
  Int32,
};
std::ostream& operator<<(std::ostream& os, ProgramUniformVariableDataType);

constexpr size_t ProgramUniformVariableDataTypeSize[] = {sizeof(float), sizeof(uint16_t), sizeof(uint32_t), sizeof(int32_t)};

constexpr std::string_view ProgramUniformVariableDataTypeName[] = {"f32", "f16", "u32", "i32"};

// represents a runtime value of a uniform variable
struct ProgramUniformVariableValue {
  ProgramUniformVariableValue();  // representing an empty uniform variable
  ProgramUniformVariableValue(float value);
  ProgramUniformVariableValue(uint32_t value);
  ProgramUniformVariableValue(int32_t value);
  ProgramUniformVariableValue(MLFloat16 value);
  ProgramUniformVariableValue(gsl::span<const float> values);
  ProgramUniformVariableValue(gsl::span<const uint32_t> values);
  ProgramUniformVariableValue(gsl::span<const int32_t> values);
  ProgramUniformVariableValue(gsl::span<const MLFloat16> values);

  size_t length;
  ProgramUniformVariableDataType data_type;
  std::vector<uint8_t> data;

 private:
  ProgramUniformVariableValue(ProgramUniformVariableDataType data_type, const void* ptr, size_t element_byte_size, size_t length = 1);
};

// represents a uniform variable definition
struct ProgramUniformVariableDefinition {
  constexpr ProgramUniformVariableDefinition(std::string_view name, ProgramUniformVariableDataType data_type)
      : name{name}, data_type{data_type} {}

  std::string_view name;
  ProgramUniformVariableDataType data_type;
};

// data type of constant
enum class ProgramConstantDataType {
  Float32,
  Float16,
  Uint32,
  Int32,
  Bool
};
std::ostream& operator<<(std::ostream& os, ProgramConstantDataType);

constexpr std::string_view ProgramConstantDataTypeName[] = {"f32", "f16", "u32", "i32", "bool"};

// represents a constant in a program
struct ProgramConstant {
  constexpr ProgramConstant(std::string_view name, float value) : name{name}, type{ProgramConstantDataType::Float32}, f32{value} {}
  constexpr ProgramConstant(std::string_view name, uint32_t value) : name{name}, type{ProgramConstantDataType::Uint32}, u32{value} {}
  constexpr ProgramConstant(std::string_view name, int32_t value) : name{name}, type{ProgramConstantDataType::Int32}, i32{value} {}
  constexpr ProgramConstant(std::string_view name, MLFloat16 value) : name{name}, type{ProgramConstantDataType::Float16}, f16{value} {}
  constexpr ProgramConstant(std::string_view name, bool value) : name{name}, type{ProgramConstantDataType::Bool}, boolean{value} {}

  std::string_view name;
  ProgramConstantDataType type;
  union {
    float f32;
    uint32_t u32;
    int32_t i32;
    MLFloat16 f16;
    bool boolean;
  };
};

// represents a runtime value of an overridable constant
struct ProgramOverridableConstantValue {
  constexpr ProgramOverridableConstantValue() : type{}, u32{}, has_value{false} {}  // representing not overriding
  constexpr ProgramOverridableConstantValue(float value) : type{ProgramConstantDataType::Float32}, f32{value}, has_value{true} {}
  constexpr ProgramOverridableConstantValue(uint32_t value) : type{ProgramConstantDataType::Uint32}, u32{value}, has_value{true} {}
  constexpr ProgramOverridableConstantValue(int32_t value) : type{ProgramConstantDataType::Int32}, i32{value}, has_value{true} {}
  constexpr ProgramOverridableConstantValue(MLFloat16 value) : type{ProgramConstantDataType::Float16}, f16{value}, has_value{true} {}
  constexpr ProgramOverridableConstantValue(bool value) : type{ProgramConstantDataType::Bool}, boolean{value}, has_value{true} {}

  ProgramConstantDataType type;
  union {
    float f32;
    uint32_t u32;
    int32_t i32;
    MLFloat16 f16;
    bool boolean;
  };
  bool has_value;
};

// represents an overridable constant definition. may or may not have a default value.
struct ProgramOverridableConstantDefinition {
  constexpr ProgramOverridableConstantDefinition(std::string_view name, ProgramConstantDataType type)
      : name{name}, type{type}, u32{}, has_default_value{false} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, float value)
      : name{name}, type{ProgramConstantDataType::Float32}, f32{value}, has_default_value{true} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, uint32_t value)
      : name{name}, type{ProgramConstantDataType::Uint32}, u32{value}, has_default_value{true} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, int32_t value)
      : name{name}, type{ProgramConstantDataType::Int32}, i32{value}, has_default_value{true} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, MLFloat16 value)
      : name{name}, type{ProgramConstantDataType::Float16}, f16{value}, has_default_value{true} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, bool value)
      : name{name}, type{ProgramConstantDataType::Bool}, boolean{value}, has_default_value{true} {}

  std::string_view name;
  ProgramConstantDataType type;
  union {
    float f32;
    uint32_t u32;
    int32_t i32;
    MLFloat16 f16;
    bool boolean;
  };
  bool has_default_value;
};

// represents whether the program shader depends on the type, rank, or shape of an input/output tensor
enum class ProgramTensorMetadataDependency : int {
  None = 0,
  Type = 1,
  Rank = 2,
  Shape = 4,
  TypeAndRank = Type | Rank,
  TypeAndShape = Type | Shape,
};
std::ostream& operator<<(std::ostream& os, ProgramTensorMetadataDependency);

inline ProgramTensorMetadataDependency operator|(ProgramTensorMetadataDependency a, ProgramTensorMetadataDependency b) {
  return (ProgramTensorMetadataDependency)((int&)a | (int&)b);
}
inline ProgramTensorMetadataDependency operator&(ProgramTensorMetadataDependency a, ProgramTensorMetadataDependency b) {
  return (ProgramTensorMetadataDependency)((int&)a & (int&)b);
}
inline ProgramTensorMetadataDependency& operator|=(ProgramTensorMetadataDependency& a, ProgramTensorMetadataDependency b) {
  return (ProgramTensorMetadataDependency&)((int&)a |= (int&)b);
}
inline ProgramTensorMetadataDependency& operator&=(ProgramTensorMetadataDependency& a, ProgramTensorMetadataDependency b) {
  return (ProgramTensorMetadataDependency&)((int&)a &= (int&)b);
}

constexpr SafeInt<uint32_t> WORKGROUP_SIZE = 64;

// data type of variable
//
// this is not a full list of all possible data types in shader programs.
// it only includes what are used in WebGPU EP.
enum class ProgramVariableDataType {
  InvalidType = -1,
  Float32,
  Vec2Float32,
  Vec4Float32,
  Float16,
  Vec2Float16,
  Vec4Float16,
  Int32,
  Vec2Int32,
  Vec4Int32,
  Uint32,
  Vec2Uint32,
  Vec4Uint32,
  Int64,
  Uint64,
  Vec4Bool,
};
#ifndef NDEBUG
std::ostream& operator<<(std::ostream& os, ProgramVariableDataType);
#endif

int NumberOfComponents(ProgramVariableDataType type);

ProgramVariableDataType ToProgramVariableDataType(int32_t element_type, int component = 1);

struct ProgramInput {
  ProgramInput(const Tensor* tensor);
  ProgramInput(const Tensor* tensor, ProgramTensorMetadataDependency dependency, int component = 1);
  ProgramInput(const Tensor* tensor, ProgramTensorMetadataDependency dependency, const TensorShape& override_shape, int component);

  const Tensor* tensor;
  ProgramTensorMetadataDependency dependency;
  ProgramVariableDataType var_type;
  bool use_override_shape;
  TensorShape override_shape;
};

struct ProgramOutput {
  ProgramOutput(Tensor* tensor);
  ProgramOutput(Tensor* tensor, ProgramTensorMetadataDependency dependency, int component = 1);
  ProgramOutput(Tensor* tensor, ProgramTensorMetadataDependency dependency, const TensorShape& override_shape, int component);

  Tensor* tensor;
  ProgramTensorMetadataDependency dependency;
  ProgramVariableDataType var_type;
  bool use_override_shape;
  TensorShape override_shape;
};

enum class ValidationMode {
  Disabled = 0,
  WGPUOnly,
  Basic,
  Full
};

namespace detail {
class ProgramWrapper;
}

struct ProgramMetadata {
  gsl::span<const ProgramConstant> constants;
  gsl::span<const ProgramOverridableConstantDefinition> overridable_constants;
  gsl::span<const ProgramUniformVariableDefinition> uniform_variables;
};

class ProgramBase {
 public:
  //
  // chain-style methods for setting properties
  //

  // set the cache hint for the program
  template <typename T>
  ProgramBase& CacheHint(T&& hint) {
    cache_hint_ = std::forward<T>(hint);
    return *this;
  }

  // add a program input
  ProgramBase& AddInput(ProgramInput&& input);
  // add multiple program inputs
  ProgramBase& AddInputs(std::initializer_list<ProgramInput> inputs);
  // add a program output
  ProgramBase& AddOutput(ProgramOutput&& output);
  // add multiple program outputs
  ProgramBase& AddOutputs(std::initializer_list<ProgramOutput> outputs);
  // add a program variable for indices
  ProgramBase& AddIndices(const TensorShape& shape);
  // add a program variable for indices
  ProgramBase& AddIndices(TensorShape&& shape);

  // set the size of dispatch groups. Y and Z are 1 if not specified.
  ProgramBase& SetDispatchGroupSize(uint32_t x);
  // set the size of dispatch groups. Z is 1 if not specified.
  ProgramBase& SetDispatchGroupSize(uint32_t x, uint32_t y);
  // set the size of dispatch groups.
  ProgramBase& SetDispatchGroupSize(uint32_t x, uint32_t y, uint32_t z);

  // set the size of a workgroup grid. Y and Z are 1 if not specified.
  ProgramBase& SetWorkgroupSize(uint32_t x);
  // set the size of a workgroup grid. Z is 1 if not specified.
  ProgramBase& SetWorkgroupSize(uint32_t x, uint32_t y);
  // set the size of a workgroup grid.
  ProgramBase& SetWorkgroupSize(uint32_t x, uint32_t y, uint32_t z);

  // add a uniform variable.
  //
  // the specified uniform variable should match the uniform definition in the class,
  // specified by macro WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES.
  ProgramBase& AddUniformVariable(ProgramUniformVariableValue&& variable);
  // add multiple uniform variables.
  //
  // the specified uniform variables should match the uniform definition in the class,
  // specified by macro WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES.
  ProgramBase& AddUniformVariables(std::initializer_list<ProgramUniformVariableValue> variables);

  // set the overridable constants
  //
  // the specified overridable constants should match the overridable constant definition in the class,
  // specified by macro WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS.
  ProgramBase& SetOverridableConstants(std::initializer_list<ProgramOverridableConstantValue> overridable_constants);

  //
  // shader code generation
  //

  virtual Status GenerateShaderCode(ShaderHelper& shader) const = 0;

  explicit ProgramBase(const std::string& name, ProgramMetadata&& metadata);

  //
  // Properties Getters
  //

  inline const std::string& Name() const { return name_; }
  inline const ProgramMetadata& Metadata() const { return metadata_; }
  inline const std::string& CacheHint() const { return cache_hint_; }
  inline const std::vector<ProgramInput>& Inputs() const { return inputs_; }
  inline const std::vector<ProgramOutput>& Outputs() const { return outputs_; }
  inline const std::vector<TensorShape>& Indices() const { return indices_; }
  inline uint32_t DispatchGroupSizeX() const { return dispatch_group_size_x_; }
  inline uint32_t DispatchGroupSizeY() const { return dispatch_group_size_y_; }
  inline uint32_t DispatchGroupSizeZ() const { return dispatch_group_size_z_; }
  inline uint32_t WorkgroupSizeX() const { return workgroup_size_x_; }
  inline uint32_t WorkgroupSizeY() const { return workgroup_size_y_; }
  inline uint32_t WorkgroupSizeZ() const { return workgroup_size_z_; }
  inline const std::vector<ProgramUniformVariableValue>& UniformVariables() const { return variables_; }
  inline const std::vector<ProgramOverridableConstantValue>& OverridableConstants() const { return overridable_constants_; }

 protected:
  virtual ~ProgramBase() = default;

 private:
  std::string name_;
  ProgramMetadata metadata_;

  std::string cache_hint_;
  std::vector<ProgramInput> inputs_;
  std::vector<ProgramOutput> outputs_;
  std::vector<TensorShape> indices_;

  uint32_t dispatch_group_size_x_;
  uint32_t dispatch_group_size_y_;
  uint32_t dispatch_group_size_z_;

  uint32_t workgroup_size_x_;
  uint32_t workgroup_size_y_;
  uint32_t workgroup_size_z_;

  std::vector<ProgramUniformVariableValue> variables_;
  std::vector<ProgramOverridableConstantValue> overridable_constants_;
};

using Program = ProgramBase;

#define WEBGPU_PROGRAM_DEFINE_(identifier, T, ...)             \
  static constexpr const T identifier##_own[] = {__VA_ARGS__}; \
  static constexpr const auto identifier =                     \
      onnxruntime::webgpu::detail::_to_std_array(identifier##_own)

#define WEBGPU_PROGRAM_EXTEND_(identifier, T, BASE, ...)       \
  static constexpr const T identifier##_own[] = {__VA_ARGS__}; \
  static constexpr const auto identifier =                     \
      onnxruntime::webgpu::detail::_concat2(BASE::identifier, identifier##_own)

#define WEBGPU_PROGRAM_DEFINE_CONSTANTS(...) \
  WEBGPU_PROGRAM_DEFINE_(constants, onnxruntime::webgpu::ProgramConstant, __VA_ARGS__)

#define WEBGPU_PROGRAM_EXTEND_CONSTANTS(BASE, ...) \
  WEBGPU_PROGRAM_EXTEND_(constants, onnxruntime::webgpu::ProgramConstant, BASE, __VA_ARGS__)

#define WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS(...) \
  WEBGPU_PROGRAM_DEFINE_(overridable_constants, onnxruntime::webgpu::ProgramOverridableConstantDefinition, __VA_ARGS__)

#define WEBGPU_PROGRAM_EXTEND_OVERRIDABLE_CONSTANTS(BASE, ...) \
  WEBGPU_PROGRAM_EXTEND_(overridable_constants, onnxruntime::webgpu::ProgramOverridableConstantDefinition, BASE, __VA_ARGS__)

#define WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(...) \
  WEBGPU_PROGRAM_DEFINE_(uniform_variables, onnxruntime::webgpu::ProgramUniformVariableDefinition, __VA_ARGS__)

#define WEBGPU_PROGRAM_EXTEND_UNIFORM_VARIABLES(BASE, ...) \
  WEBGPU_PROGRAM_EXTEND_(uniform_variables, onnxruntime::webgpu::ProgramUniformVariableDefinition, BASE, __VA_ARGS__)

}  // namespace webgpu
}  // namespace onnxruntime
