#ifdef _WIN32
#include <Windows.h>
#else
#include <wsl/winadapter.h>
#endif

#include <wil/result.h>
#include <wrl/client.h>
#include <iostream>
#include "DxDispatchInterface.h"

using namespace Microsoft::WRL;

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
    if (hr == S_OK)
    {
        RETURN_IF_FAILED(dispatch->RunAll());
    }
    else
    {
        // ignores S_FALSE
        if (FAILED(hr))
        {
            printf("%s failed with hr=0x%08x\n",argv[0], hr);
        }
    }

    return hr;
}