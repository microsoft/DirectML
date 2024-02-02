#include "pch.h"
#include "TensorHelper.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <onnxruntime_cxx_api.h>
#include "dml_provider_factory.h"

inline void ThrowOnFailed(HRESULT hr) {
  if (FAILED(hr)) {
      throw;
  }
}

using UniqueNativePtr = std::unique_ptr<void, void (*)(void*)>;

size_t GetSizeFromType(ONNXTensorElementDataType type) {
#define CASE_FOR_TYPE(T)                 \
  case Ort::TypeToTensorType<T>::type: { \
    return sizeof(T);                    \
  }

  switch (type) {
    CASE_FOR_TYPE(Ort::Float16_t);
    CASE_FOR_TYPE(Ort::BFloat16_t);
    CASE_FOR_TYPE(float);
    CASE_FOR_TYPE(double);
    CASE_FOR_TYPE(int8_t);
    CASE_FOR_TYPE(int16_t);
    CASE_FOR_TYPE(int32_t);
    CASE_FOR_TYPE(int64_t);
    CASE_FOR_TYPE(uint8_t);
    CASE_FOR_TYPE(uint16_t);
    CASE_FOR_TYPE(uint32_t);
    CASE_FOR_TYPE(uint64_t);
    CASE_FOR_TYPE(bool);
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_FOR_TYPE(Ort::Float8E4M3FN_t);
    CASE_FOR_TYPE(Ort::Float8E4M3FNUZ_t);
    CASE_FOR_TYPE(Ort::Float8E5M2_t);
    CASE_FOR_TYPE(Ort::Float8E5M2FNUZ_t);
#endif
    default:
        throw;
  }
#undef CASE_FOR_TYPE
}

Microsoft::WRL::ComPtr<ID3D12Resource> CreateD3D12Resource(
    ID3D12Device* device,
    ONNXTensorElementDataType type,
    const std::vector<int64_t>& shape,
    D3D12_HEAP_TYPE heap_type) {
  // Try to allocate the backing memory for the caller
  auto bufferSize =
      std::accumulate(
          std::begin(shape),
          std::end(shape),
          static_cast<int64_t>(1),
          std::multiplies<int64_t>());

  auto bufferByteSize = GetSizeFromType(type) * bufferSize;

  // DML needs the resources' sizes to be a multiple of 4 bytes
  if (bufferByteSize % 4 != 0) {
    bufferByteSize += 4 - (bufferByteSize % 4);
  }

  auto resource_flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  if (heap_type == D3D12_HEAP_TYPE_UPLOAD ||
      heap_type == D3D12_HEAP_TYPE_READBACK) {
    resource_flags = D3D12_RESOURCE_FLAG_NONE;
  }

  D3D12_HEAP_PROPERTIES heapProperties = {
      heap_type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0};
  D3D12_RESOURCE_DESC resourceDesc = {
      D3D12_RESOURCE_DIMENSION_BUFFER,
      0,
      static_cast<uint64_t>(bufferByteSize),
      1,
      1,
      1,
      DXGI_FORMAT_UNKNOWN,
      {1, 0},
      D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      resource_flags};

  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  ThrowOnFailed(device->CreateCommittedResource(
      &heapProperties,
      D3D12_HEAP_FLAG_NONE,
      &resourceDesc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      __uuidof(ID3D12Resource),
      &resource));

  return resource;
}

std::pair<Ort::Value, UniqueNativePtr> CreateDmlValue(
    const Ort::ConstTensorTypeAndShapeInfo& tensor_info,
    ID3D12CommandQueue* queue) {

  auto& ortApi = Ort::GetApi();
  const OrtDmlApi* ortDmlApi;
  Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));

  Microsoft::WRL::ComPtr<ID3D12Device> device = nullptr;
  ThrowOnFailed(queue->GetDevice(IID_PPV_ARGS(&device)));

  auto shape = tensor_info.GetShape();
  auto resource = CreateD3D12Resource(device.Get(), tensor_info.GetElementType(), shape, D3D12_HEAP_TYPE_DEFAULT);
  void* dmlAllocatorResource;
  Ort::ThrowOnError(ortDmlApi->CreateGPUAllocationFromD3DResource(resource.Get(), &dmlAllocatorResource));

  auto uniqueDmlAllocatorResource = UniqueNativePtr(dmlAllocatorResource, [](void* ptr) {
    auto& ortApi = Ort::GetApi();
    const OrtDmlApi* ortDmlApi;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
    Ort::ThrowOnError(ortDmlApi->FreeGPUAllocation(ptr));
  });

  auto memoryInfo = Ort::MemoryInfo("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);

  // create the OrtValue as a tensor letting ort know that we own the data buffer
  OrtValue* value;
  Ort::ThrowOnError(ortApi.CreateTensorWithDataAsOrtValue(
      memoryInfo,
      uniqueDmlAllocatorResource.get(),
      static_cast<size_t>(resource->GetDesc().Width),
      shape.data(),
      shape.size(),
      tensor_info.GetElementType(),
      &value));
  
  return { Ort::Value(value), std::move(uniqueDmlAllocatorResource)};
}