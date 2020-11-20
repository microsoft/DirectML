// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561

// Root signatures must match definition in each effect, or shaders will be recompiled on Xbox when PSO loads
#define NoTextureRS \
"RootFlags ( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"CBV(b0)"

#define MainRS \
"RootFlags ( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"CBV(b0),"\
"DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL ),"\
"DescriptorTable ( Sampler(s0), visibility = SHADER_VISIBILITY_PIXEL )"

#define DualTextureRS \
"RootFlags ( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL )," \
"DescriptorTable ( Sampler(s0), visibility = SHADER_VISIBILITY_PIXEL )," \
"DescriptorTable ( SRV(t1), visibility = SHADER_VISIBILITY_PIXEL )," \
"DescriptorTable ( Sampler(s1), visibility = SHADER_VISIBILITY_PIXEL )," \
"CBV(b0)"

#define NormalMapRS \
"RootFlags ( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL )," \
"DescriptorTable ( SRV(t1), visibility = SHADER_VISIBILITY_PIXEL )," \
"DescriptorTable ( Sampler(s0), visibility = SHADER_VISIBILITY_PIXEL )," \
"CBV(b0)," \
"DescriptorTable ( SRV(t2), visibility = SHADER_VISIBILITY_PIXEL )"

#define NormalMapRSNoSpec \
"RootFlags ( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL )," \
"DescriptorTable ( SRV(t1), visibility = SHADER_VISIBILITY_PIXEL )," \
"DescriptorTable ( Sampler(s0), visibility = SHADER_VISIBILITY_PIXEL )," \
"CBV(b0)"

#define GenerateMipsRS \
"RootFlags ( DENY_VERTEX_SHADER_ROOT_ACCESS   |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS   |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS     |" \
"            DENY_PIXEL_SHADER_ROOT_ACCESS )," \
"RootConstants(num32BitConstants=3, b0)," \
"DescriptorTable ( SRV(t0) )," \
"DescriptorTable ( UAV(u0) )," \
"StaticSampler(s0,"\
"           filter =   FILTER_MIN_MAG_LINEAR_MIP_POINT,"\
"           addressU = TEXTURE_ADDRESS_CLAMP,"\
"           addressV = TEXTURE_ADDRESS_CLAMP,"\
"           addressW = TEXTURE_ADDRESS_CLAMP )"

#define SpriteStaticRS \
"RootFlags ( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL ),"\
"CBV(b0), "\
"StaticSampler(s0,"\
"           filter = FILTER_MIN_MAG_MIP_LINEAR,"\
"           addressU = TEXTURE_ADDRESS_CLAMP,"\
"           addressV = TEXTURE_ADDRESS_CLAMP,"\
"           addressW = TEXTURE_ADDRESS_CLAMP,"\
"           visibility = SHADER_VISIBILITY_PIXEL )"

#define SpriteHeapRS \
"RootFlags ( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL ),"\
"CBV(b0), " \
"DescriptorTable ( Sampler(s0), visibility = SHADER_VISIBILITY_PIXEL )"

#define PostProcessRS \
"RootFlags ( DENY_VERTEX_SHADER_ROOT_ACCESS |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL ),"\
"CBV(b0, visibility = SHADER_VISIBILITY_PIXEL ), " \
"StaticSampler(s0,"\
"           filter =   FILTER_MIN_MAG_MIP_LINEAR,"\
"           addressU = TEXTURE_ADDRESS_CLAMP,"\
"           addressV = TEXTURE_ADDRESS_CLAMP,"\
"           addressW = TEXTURE_ADDRESS_CLAMP,"\
"           visibility = SHADER_VISIBILITY_PIXEL )"

#define PostProcessRSNoCB \
"RootFlags ( DENY_VERTEX_SHADER_ROOT_ACCESS |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL ),"\
"StaticSampler(s0,"\
"           filter =   FILTER_MIN_MAG_MIP_LINEAR,"\
"           addressU = TEXTURE_ADDRESS_CLAMP,"\
"           addressV = TEXTURE_ADDRESS_CLAMP,"\
"           addressW = TEXTURE_ADDRESS_CLAMP,"\
"           visibility = SHADER_VISIBILITY_PIXEL )"

#define DualPostProcessRS \
"RootFlags ( DENY_VERTEX_SHADER_ROOT_ACCESS |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL ),"\
"DescriptorTable ( SRV(t1), visibility = SHADER_VISIBILITY_PIXEL ),"\
"CBV(b0, visibility = SHADER_VISIBILITY_PIXEL ), " \
"StaticSampler(s0,"\
"           filter =   FILTER_MIN_MAG_MIP_LINEAR,"\
"           addressU = TEXTURE_ADDRESS_CLAMP,"\
"           addressV = TEXTURE_ADDRESS_CLAMP,"\
"           addressW = TEXTURE_ADDRESS_CLAMP,"\
"           visibility = SHADER_VISIBILITY_PIXEL )"

#define ToneMapRS \
"RootFlags ( DENY_VERTEX_SHADER_ROOT_ACCESS |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL ),"\
"CBV(b0, visibility = SHADER_VISIBILITY_PIXEL ), " \
"StaticSampler(s0,"\
"           filter =   FILTER_MIN_MAG_MIP_POINT,"\
"           addressU = TEXTURE_ADDRESS_CLAMP,"\
"           addressV = TEXTURE_ADDRESS_CLAMP,"\
"           addressW = TEXTURE_ADDRESS_CLAMP,"\
"           visibility = SHADER_VISIBILITY_PIXEL )"

#define PBREffectRS \
"RootFlags ( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"DescriptorTable ( SRV(t0) ),"\
"DescriptorTable ( SRV(t1) ),"\
"DescriptorTable ( SRV(t2) ),"\
"DescriptorTable ( SRV(t3) ),"\
"DescriptorTable ( SRV(t4) ),"\
"DescriptorTable ( SRV(t5) ),"\
"DescriptorTable ( Sampler(s0) ),"\
"DescriptorTable ( Sampler(s1) ),"\
"CBV(b0)"

#define DebugEffectRS \
"RootFlags ( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |" \
"            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
"            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
"            DENY_HULL_SHADER_ROOT_ACCESS )," \
"CBV(b0)"
