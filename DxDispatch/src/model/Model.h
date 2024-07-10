#pragma once

#include <filesystem>
#include <unordered_map>
#include <optional>
#include <variant>
#include <string>
#include <vector>
#include <gsl/gsl>
#include <DirectML.h>
#include "BucketAllocator.h"

enum class DmlCompileType 
{
    DmlCompileOp,
    DmlCompileGraph
};

class Model
{
public:
    // When binding a buffer to an operator it is possible to use a subregion of
    // the buffer by specifying an elementOffset, elementCount, and elementSizeInBytes.
    // Additionally, an optional format specifier dictates how to interpret the buffer
    // contents; when omitted the buffer will be interpreted using the same data type used
    // to initialize it.
    struct BufferBindingSource
    {
        std::string name;
        uint64_t elementCount;
        uint64_t elementSizeInBytes;
        uint64_t elementOffset;
        std::optional<DXGI_FORMAT> format;

        // For Append/Consume buffers only:
        std::optional<std::string> counterName;
        uint64_t counterOffsetBytes;

        std::vector<int64_t> shape;
    };

    using Bindings = std::unordered_map<std::string, std::vector<BufferBindingSource>>;

    // RESOURCES
    // ------------------------------------------------------------------------

    struct BufferDesc
    {
        uint64_t sizeInBytes;
        std::vector<std::byte> initialValues;
        DML_TENSOR_DATA_TYPE initialValuesDataType;
        uint64_t initialValuesOffsetInBytes;
        bool useDeferredBinding;
    };

    struct ResourceDesc
    {
        std::string name;
        std::variant<BufferDesc> value;
    };

    // DISPATCHABLES
    // ------------------------------------------------------------------------

    struct DmlDispatchableDesc
    {
        struct BindPoint
        {
            std::string name;
            uint32_t resourceCount;
            bool required;
            bool requiredBinding;
        };

        struct BindPoints
        {
            std::vector<BindPoint> inputs;
            std::vector<BindPoint> outputs;
        };

        DML_OPERATOR_DESC* desc;
        BindPoints bindPoints;
        DML_EXECUTION_FLAGS executionFlags;
        DmlCompileType compileType;
        Bindings initBindings;
    };

    struct HlslDispatchableDesc
    {
        enum class Compiler
        {
            DXC
        };

        std::filesystem::path sourcePath;
        Compiler compiler;
        std::vector<std::string> compilerArgs;
    };

    struct OnnxDispatchableDesc
    {
        std::filesystem::path sourcePath;

        // NOTE: these will be overriden if also using the respective command-line options.
        std::vector<std::pair<std::string, uint32_t>> freeDimNameOverrides;
        std::vector<std::pair<std::string, uint32_t>> freeDimDenotationOverrides;
        std::vector<std::pair<std::string, std::string>> sessionOptionsConfigEntries;
        uint32_t graphOptimizationLevel = 99;
        uint32_t loggingLevel = 2;
    };

    struct DispatchableDesc
    {
        std::string name;
        std::variant<DmlDispatchableDesc, HlslDispatchableDesc, OnnxDispatchableDesc> value;
    };

    // COMMANDS
    // ------------------------------------------------------------------------

    struct DispatchCommand
    {
        std::string dispatchableName;
        Bindings bindings;
        std::array<uint32_t, 3> threadGroupCount;
    };

    struct PrintCommand
    {
        std::string resourceName;
    };

    struct WriteFileCommand
    {
        std::string resourceName;
        std::string targetPath;
        std::vector<uint32_t> dimensions; // The resources don't store their dimensions. So repeat them here.
    };

    using Command = std::variant<DispatchCommand, PrintCommand, WriteFileCommand>;

    struct CommandDesc
    {
        std::string type;
        std::string parameters;
        Command command;
    };

    Model() = default;

    Model(
        std::vector<ResourceDesc>&& resourceDescs,
        std::vector<DispatchableDesc>&& dispatchableDescs,
        std::vector<CommandDesc>&& commands,
        BucketAllocator&& allocator);

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    gsl::span<const ResourceDesc> GetResourceDescs() const { return m_resourceDescs; }
    gsl::span<const DispatchableDesc> GetDispatchableDescs() const { return m_dispatchableDescs; }
    gsl::span<const CommandDesc> GetCommands() const { return m_commands; }

    const ResourceDesc& GetResource(std::string_view name) const { return *m_resourceDescsByName.find(name.data())->second; }
    const DispatchableDesc& GetDispatchable(std::string_view name) const { return *m_dispatchableDescsByName.find(name.data())->second; }

private:
    std::vector<ResourceDesc> m_resourceDescs;
    std::vector<DispatchableDesc> m_dispatchableDescs;
    std::vector<CommandDesc> m_commands;
    BucketAllocator m_allocator;
    std::unordered_map<std::string, ResourceDesc*> m_resourceDescsByName;
    std::unordered_map<std::string, DispatchableDesc*> m_dispatchableDescsByName;
};