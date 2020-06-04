//--------------------------------------------------------------------------------------
// File: Model.h
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#pragma once

#if defined(_XBOX_ONE) && defined(_TITLE)
#include <d3d12_x.h>
#else
#include <d3d12.h>
#endif

#include <DirectXMath.h>
#include <DirectXCollision.h>

#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <assert.h>
#include <stdint.h>

#include "GraphicsMemory.h"
#include "Effects.h"


namespace DirectX
{
    class IEffect;
    class IEffectFactory;
    class ModelMesh;

    //----------------------------------------------------------------------------------
    // Each mesh part is a submesh with a single effect
    class ModelMeshPart
    {
    public:
        ModelMeshPart(uint32_t partIndex);
        virtual ~ModelMeshPart();

        uint32_t                                                partIndex;      // Unique index assigned per-part in a model; used to index effects.
        uint32_t                                                materialIndex;  // Index of the material spec to use
        uint32_t                                                indexCount;
        uint32_t                                                startIndex;
        int32_t                                                 vertexOffset;
        uint32_t                                                vertexStride;
        uint32_t                                                vertexCount;
        uint32_t                                                indexBufferSize;
        uint32_t                                                vertexBufferSize;
        D3D_PRIMITIVE_TOPOLOGY                                  primitiveType;
        DXGI_FORMAT                                             indexFormat;
        SharedGraphicsResource                                  indexBuffer;
        SharedGraphicsResource                                  vertexBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource>                  staticIndexBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource>                  staticVertexBuffer;
        std::shared_ptr<std::vector<D3D12_INPUT_ELEMENT_DESC>>  vbDecl;

        using Collection = std::vector<std::unique_ptr<ModelMeshPart>>;
        using DrawCallback = std::function<void(_In_ ID3D12GraphicsCommandList* commandList, _In_ const ModelMeshPart& part)>;

        // Draw mesh part
        void __cdecl Draw(_In_ ID3D12GraphicsCommandList* commandList) const;

        void __cdecl DrawInstanced(_In_ ID3D12GraphicsCommandList* commandList, uint32_t instanceCount, uint32_t startInstanceLocation = 0) const;

        //
        // Utilities for drawing multiple mesh parts
        //

        // Draw the mesh
        static void __cdecl DrawMeshParts(_In_ ID3D12GraphicsCommandList* commandList, _In_ const ModelMeshPart::Collection& meshParts);

        // Draw the mesh with an effect
        static void __cdecl DrawMeshParts(_In_ ID3D12GraphicsCommandList* commandList, _In_ const ModelMeshPart::Collection& meshParts, _In_ IEffect* effect);

        // Draw the mesh with a callback for each mesh part
        static void __cdecl DrawMeshParts(_In_ ID3D12GraphicsCommandList* commandList, _In_ const ModelMeshPart::Collection& meshParts, DrawCallback callback);

        // Draw the mesh with a range of effects that mesh parts will index into. 
        // Effects can be any IEffect pointer type (including smart pointer). Value or reference types will not compile.
        // The iterator passed to this method should have random access capabilities for best performance.
        template<typename TEffectIterator, typename TEffectIteratorCategory = typename TEffectIterator::iterator_category>
        static void DrawMeshParts(
            _In_ ID3D12GraphicsCommandList* commandList,
            _In_ const ModelMeshPart::Collection& meshParts,
            TEffectIterator partEffects)
        {
            // This assert is here to prevent accidental use of containers that would cause undesirable performance penalties.
            static_assert(
                std::is_base_of<std::random_access_iterator_tag, TEffectIteratorCategory>::value,
                "Providing an iterator without random access capabilities -- such as from std::list -- is not supported.");

            for (auto it = std::begin(meshParts); it != std::end(meshParts); ++it)
            {
                auto part = it->get();
                assert(part != nullptr);

                // Get the effect at the location specified by the part's material
                TEffectIterator effect_iterator = partEffects;
                std::advance(effect_iterator, part->partIndex);

                // Apply the effect and draw
                (*effect_iterator)->Apply(commandList);
                part->Draw(commandList);
            }
        }
    };


    //----------------------------------------------------------------------------------
    // A mesh consists of one or more model mesh parts
    class ModelMesh
    {
    public:
        ModelMesh() noexcept;
        virtual ~ModelMesh();

        BoundingSphere              boundingSphere;
        BoundingBox                 boundingBox;
        ModelMeshPart::Collection   opaqueMeshParts;
        ModelMeshPart::Collection   alphaMeshParts;
        std::wstring                name;

        using Collection = std::vector<std::shared_ptr<ModelMesh>>;

        // Draw the mesh
        void __cdecl DrawOpaque(_In_ ID3D12GraphicsCommandList* commandList) const;
        void __cdecl DrawAlpha(_In_ ID3D12GraphicsCommandList* commandList) const;

        // Draw the mesh with an effect
        void __cdecl DrawOpaque(_In_ ID3D12GraphicsCommandList* commandList, _In_ IEffect* effect) const;
        void __cdecl DrawAlpha(_In_ ID3D12GraphicsCommandList* commandList, _In_ IEffect* effect) const;

        // Draw the mesh with a callback for each mesh part
        void __cdecl DrawOpaque(_In_ ID3D12GraphicsCommandList* commandList, ModelMeshPart::DrawCallback callback) const;
        void __cdecl DrawAlpha(_In_ ID3D12GraphicsCommandList* commandList, ModelMeshPart::DrawCallback callback) const;

        // Draw the mesh with a range of effects that mesh parts will index into. 
        // TEffectPtr can be any IEffect pointer type (including smart pointer). Value or reference types will not compile.
        template<typename TEffectIterator, typename TEffectIteratorCategory = typename TEffectIterator::iterator_category>
        void DrawOpaque(_In_ ID3D12GraphicsCommandList* commandList, TEffectIterator effects) const
        {
            ModelMeshPart::DrawMeshParts<TEffectIterator, TEffectIteratorCategory>(commandList, opaqueMeshParts, effects);
        }
        template<typename TEffectIterator, typename TEffectIteratorCategory = typename TEffectIterator::iterator_category>
        void DrawAlpha(_In_ ID3D12GraphicsCommandList* commandList, TEffectIterator effects) const
        {
            ModelMeshPart::DrawMeshParts<TEffectIterator, TEffectIteratorCategory>(commandList, alphaMeshParts, effects);
        }
    };


    //----------------------------------------------------------------------------------
    // A model consists of one or more meshes
    class Model
    {
    public:
        Model() noexcept;
        virtual ~Model();

        using ModelMaterialInfo = IEffectFactory::EffectInfo;
        using ModelMaterialInfoCollection = std::vector<ModelMaterialInfo>;
        using TextureCollection = std::vector<std::wstring>;

        //
        // NOTE
        // 
        // The Model::Draw functions use variadic templates and perfect-forwarding in order to support future overloads to the ModelMesh::Draw
        // family of functions. This means that a new ModelMesh::Draw overload can be added, removed or altered, but the Model::Draw* routines
        // will still remain compatible. The correct ModelMesh::Draw overload will be selected by the compiler depending on the arguments you 
        // provide to Model::Draw*.
        //

        // Draw all the opaque meshes in the model
        template<typename... TForwardArgs> void DrawOpaque(_In_ ID3D12GraphicsCommandList* commandList, TForwardArgs&&... args) const
        {
            // Draw opaque parts
            for (auto it = std::begin(meshes); it != std::end(meshes); ++it)
            {
                auto mesh = it->get();
                assert(mesh != nullptr);

                mesh->DrawOpaque(commandList, std::forward<TForwardArgs>(args)...);
            }
        }

        // Draw all the alpha meshes in the model
        template<typename... TForwardArgs> void DrawAlpha(_In_ ID3D12GraphicsCommandList* commandList, TForwardArgs&&... args) const
        {
            // Draw opaque parts
            for (auto it = std::begin(meshes); it != std::end(meshes); ++it)
            {
                auto mesh = it->get();
                assert(mesh != nullptr);

                mesh->DrawAlpha(commandList, std::forward<TForwardArgs>(args)...);
            }
        }

        // Draw all the meshes in the model
        template<typename... TForwardArgs> void Draw(_In_ ID3D12GraphicsCommandList* commandList, TForwardArgs&&... args) const
        {
            DrawOpaque(commandList, std::forward<TForwardArgs>(args)...);
            DrawAlpha(commandList, std::forward<TForwardArgs>(args)...);
        }

        // Load texture resources into an existing Effect Texture Factory
        int __cdecl LoadTextures(IEffectTextureFactory& texFactory, int destinationDescriptorOffset = 0) const;

        // Load texture resources into a new Effect Texture Factory
        std::unique_ptr<EffectTextureFactory> __cdecl LoadTextures(
            _In_ ID3D12Device* device,
            ResourceUploadBatch& resourceUploadBatch,
            _In_opt_z_ const wchar_t* texturesPath = nullptr,
            D3D12_DESCRIPTOR_HEAP_FLAGS flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) const;

        // Load VB/IB resources for static geometry
        void __cdecl LoadStaticBuffers(
            _In_ ID3D12Device* device,
            ResourceUploadBatch& resourceUploadBatch,
            bool keepMemory = false);

        // Create effects using the default effect factory
        std::vector<std::shared_ptr<IEffect>> __cdecl CreateEffects(
            const EffectPipelineStateDescription& opaquePipelineState,
            const EffectPipelineStateDescription& alphaPipelineState,
            _In_ ID3D12DescriptorHeap* textureDescriptorHeap,
            _In_ ID3D12DescriptorHeap* samplerDescriptorHeap,
            int textureDescriptorOffset = 0,
            int samplerDescriptorOffset = 0) const;

        // Create effects using a custom effect factory
        std::vector<std::shared_ptr<IEffect>> __cdecl CreateEffects(
            IEffectFactory& fxFactory,
            const EffectPipelineStateDescription& opaquePipelineState,
            const EffectPipelineStateDescription& alphaPipelineState,
            int textureDescriptorOffset = 0,
            int samplerDescriptorOffset = 0) const;

        // Loads a model from a DirectX SDK .SDKMESH file
        static std::unique_ptr<Model> __cdecl CreateFromSDKMESH(_In_reads_bytes_(dataSize) const uint8_t* meshData, _In_ size_t dataSize, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<Model> __cdecl CreateFromSDKMESH(_In_z_ const wchar_t* szFileName, _In_opt_ ID3D12Device* device = nullptr);

        // Loads a model from a .VBO file
        static std::unique_ptr<Model> __cdecl CreateFromVBO(_In_reads_bytes_(dataSize) const uint8_t* meshData, _In_ size_t dataSize, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<Model> __cdecl CreateFromVBO(_In_z_ const wchar_t* szFileName, _In_opt_ ID3D12Device* device = nullptr);

        // Utility function for getting a GPU descriptor for a mesh part/material index. If there is no texture the 
        // descriptor will be zero.
        D3D12_GPU_DESCRIPTOR_HANDLE GetGpuTextureHandleForMaterialIndex(uint32_t materialIndex, _In_ ID3D12DescriptorHeap* heap, _In_ size_t descriptorSize, _In_ size_t descriptorOffset) const
        {
            D3D12_GPU_DESCRIPTOR_HANDLE handle = {};

            if (materialIndex >= materials.size())
                return handle;

            int textureIndex = materials[materialIndex].diffuseTextureIndex;
            if (textureIndex == -1)
                return handle;

            handle = heap->GetGPUDescriptorHandleForHeapStart();
            handle.ptr += static_cast<UINT64>(descriptorSize * (UINT64(textureIndex) + UINT64(descriptorOffset)));

            return handle;
        }

        // Utility function for updating the matrices in a list of effects. This will SetWorld, SetView and SetProjection
        // on any effect in the list that derives from IEffectMatrices.
        static void XM_CALLCONV UpdateEffectMatrices(
            _In_ std::vector<std::shared_ptr<IEffect>>& effectList,
            DirectX::FXMMATRIX world,
            DirectX::CXMMATRIX view,
            DirectX::CXMMATRIX proj);

        ModelMesh::Collection           meshes;
        ModelMaterialInfoCollection     materials;
        TextureCollection               textureNames;
        std::wstring                    name;

    private:
        std::shared_ptr<IEffect> __cdecl CreateEffectForMeshPart(
            IEffectFactory& fxFactory,
            const EffectPipelineStateDescription& opaquePipelineState,
            const EffectPipelineStateDescription& alphaPipelineState,
            int textureDescriptorOffset,
            int samplerDescriptorOffset,
            _In_ const ModelMeshPart* part) const;
    };
}
