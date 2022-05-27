#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "OnnxDispatchable.h"

using Microsoft::WRL::ComPtr;

OnnxDispatchable::OnnxDispatchable(
    std::string_view name, 
    std::shared_ptr<Device> device, 
    const Model::OnnxDispatchableDesc& desc,
    const Dispatchable::Bindings& initBindings
    ) : m_name(name), m_device(device), m_desc(desc), m_initBindings(std::move(initBindings))
{
}

void OnnxDispatchable::Initialize()
{

}

void OnnxDispatchable::Bind(const Bindings& bindings)
{

}

void OnnxDispatchable::Dispatch(const Model::DispatchCommand& args)
{
}