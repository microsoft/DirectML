#ifdef _WIN32
#include <Windows.h>
#endif

#include <wrl/client.h>
#include <wil/result.h>
#include "dxDispatchInterface.h"

using Microsoft::WRL::ComPtr;
interface IAdapter;

int main(int argc, char** argv)
{
    ComPtr<IDxDispatch> dispatch;
    HRESULT hr = S_OK;
    hr = CreateDxDispatchFromArgs(
        argc,
        argv,
        nullptr,
        nullptr,
        nullptr,    // will log to console if not overwritten
        &dispatch
    );
    if(hr == S_OK)
    {
        RETURN_IF_FAILED(dispatch->RunAll());
    }
    else
    {
        // ignores S_FALSE
        RETURN_IF_FAILED(hr);
    }

    return hr;
}