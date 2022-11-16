#include "pch.h"
#include "JsonParsers.h"
#include "StdSupport.h"

////////////////////////////////////////
// Generic helpers

inline const char* ToChar(const char8_t* p) { return reinterpret_cast<const char*>(p); }

template<
    typename ContiguousContainerType,
    typename ElementType = std::remove_reference_t<decltype(*std::declval<ContiguousContainerType>().data())>
>
auto append_data(ContiguousContainerType& v, std::span<const ElementType> s)
{
    // Whyyyy is basic functionality like std::vector::append() missing from the standard? -_-
    v.insert(v.end(), s.begin(), s.end());
}

// Reinterprets a span of data from one type to another.
// The input parameter can be any contiguous container with data() and size() methods,
// including gsl::span, std::array, and std::vector.
template <typename NewType, typename OldTypeContainer>
std::span<NewType> reinterpret_span(OldTypeContainer& oldSpan)
{
    using OldType = decltype(*oldSpan.data());
    size_t newElementCount = static_cast<size_t>(oldSpan.size()) * sizeof(OldType) / sizeof(NewType);
    assert(newElementCount * sizeof(NewType) == oldSpan.size() * sizeof(OldType));

    NewType* p = reinterpret_cast<NewType*>(oldSpan.data());
    return std::span<NewType>(p, p + newElementCount);
}

template<
    typename ContiguousContainerType,
    typename ElementType = std::remove_reference_t<decltype(*std::declval<ContiguousContainerType>().data())>
>
auto make_span(ContiguousContainerType& container) -> std::span<ElementType>
{
    auto* begin = std::data(container);
    return std::span<ElementType>(begin, begin + std::size(container));
}

template<typename ContiguousContainerType>
std::span<const std::byte> as_bytes(const ContiguousContainerType& container)
#if __cpp_concepts
requires (std::is_const_v<std::remove_pointer_t<decltype(container.data())>>)
#endif
{
    auto oldSpan = make_span(container);
    return std::span<const std::byte>(reinterpret_cast<const std::byte*>(oldSpan.data()), oldSpan.size_bytes());
}

#if __cpp_concepts
template<typename ContiguousContainerType>
std::span<std::byte> as_bytes(ContiguousContainerType& container)
requires (!std::is_const_v<std::remove_pointer_t<decltype(container.data())>>)
{
    auto oldSpan = make_span(container);
    return std::span<std::byte>(reinterpret_cast<std::byte*>(oldSpan.data()), oldSpan.size_bytes());
}
#endif

// Reads a byte array from std::vector/std::string/std::array as a struct.
template <typename NewStructType, typename OldTypeContainer>
const NewStructType& read_as(OldTypeContainer&& oldSpan)
{
    std::span<const std::byte> byteSpan = as_bytes(oldSpan);
    size_t byteSize = byteSpan.size_bytes();
    if (sizeof(NewStructType) > byteSize)
    {
        throw std::runtime_error("Span is too small to be cast to new data type.");
    }
    return *reinterpret_cast<const NewStructType*>(byteSpan.data());
}

bool IsNpyFilenameExtension(std::string_view filename)
{
    return ends_with(filename, std::string_view(".npy")) || ends_with(filename, std::string_view(".NPY"));
}

////////////////////////////////////////
// Tensor specific

constexpr uint32_t g_elementDataTypeByteSizes[] =
{
    0, // DML_TENSOR_DATA_TYPE_UNKNOWN,
    4, // DML_TENSOR_DATA_TYPE_FLOAT32,
    2, // DML_TENSOR_DATA_TYPE_FLOAT16,
    4, // DML_TENSOR_DATA_TYPE_UINT32,
    2, // DML_TENSOR_DATA_TYPE_UINT16,
    1, // DML_TENSOR_DATA_TYPE_UINT8,
    4, // DML_TENSOR_DATA_TYPE_INT32,
    2, // DML_TENSOR_DATA_TYPE_INT16,
    1, // DML_TENSOR_DATA_TYPE_INT8,
    8, // DML_TENSOR_DATA_TYPE_FLOAT64,
    8, // DML_TENSOR_DATA_TYPE_UINT64,
    8, // DML_TENSOR_DATA_TYPE_INT64,
};

uint32_t GetByteSizeFromDataType(DML_TENSOR_DATA_TYPE dataType) noexcept
{
    size_t index = static_cast<size_t>(dataType);
    return g_elementDataTypeByteSizes[index < std::size(g_elementDataTypeByteSizes) ? index : 0];
}

uint32_t ComputeElementCount(std::span<const int32_t> dimensions)
{
    return std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<int32_t>());
}

////////////////////////////////////////
// NumPy specific

void MapNumPyArrayDataTypeToDml(
    std::u8string_view numPyElementType,
    /*out*/ DML_TENSOR_DATA_TYPE& dataType,
    /*out*/ bool& isBackwardsEndian // Backwards endian which stores greatest bytes at lowest bytes.
    )
{
    dataType = DML_TENSOR_DATA_TYPE_UNKNOWN;
    isBackwardsEndian = false;

    DML_TENSOR_DATA_TYPE resolvedDataType = DML_TENSOR_DATA_TYPE_UNKNOWN;
    uint32_t elementByteSize = 0;

    #if !(defined(_M_IX86) || defined(_M_X64) || defined(_M_ARM) || defined(_M_ARM64) || (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)))
    // Technically ARM machines can accept either, but the vast majority of ARM machines
    // default to logical endian, including all 2021 Windows ones and Android phones.
    static_assert(false, "Double check that endianness is specified correctly for this architecture when using '='.");
    #endif

    // Parse the string (e.g. f, i, i4, f8) into a default data type and (if given) byte size.
    // https://docs.python.org/2/library/array.html#module-array
    // https://numpy.org/devdocs/reference/arrays.dtypes.html
    for (char c : numPyElementType)
    {
        switch (c)
        {
        case '?': resolvedDataType = DML_TENSOR_DATA_TYPE_UINT8;   break; // boolean
        case 'b': resolvedDataType = DML_TENSOR_DATA_TYPE_INT8;    break; // signed byte
        case 'B': resolvedDataType = DML_TENSOR_DATA_TYPE_UINT8;   break; // unsigned byte
        case 'h': resolvedDataType = DML_TENSOR_DATA_TYPE_INT16;   break; // signed short
        case 'H': resolvedDataType = DML_TENSOR_DATA_TYPE_UINT16;  break; // unsigned short
        case 'i': resolvedDataType = DML_TENSOR_DATA_TYPE_INT32;   break; // signed integer
        case 'u': resolvedDataType = DML_TENSOR_DATA_TYPE_UINT32;  break; // unsigned integer
        case 'f': resolvedDataType = DML_TENSOR_DATA_TYPE_FLOAT32; break; // float
        case 'd': resolvedDataType = DML_TENSOR_DATA_TYPE_FLOAT64; break; // float64
        case '>': isBackwardsEndian = true; break;    // (backwards-endian)
        case '<': isBackwardsEndian = false; break;   // (logical-endian)
        case '=': isBackwardsEndian = false; break;   // (logical-endian since targeting x86)
        case '|': isBackwardsEndian = false; break;   // not applicable

        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            elementByteSize = elementByteSize * 10 + c - '0';
            break;

        default:
            assert(false);
            resolvedDataType = DML_TENSOR_DATA_TYPE_UNKNOWN;
            break;
        }
    }

    // The second pass resolves data types if a specific size was given, such as "f8" for float64.
    // Otherwise the default type remains (e.g. "f" for float32).
    if (elementByteSize > 0)
    {
        switch (resolvedDataType)
        {
        case DML_TENSOR_DATA_TYPE_UINT8:
        case DML_TENSOR_DATA_TYPE_UINT16:
        case DML_TENSOR_DATA_TYPE_UINT32:
            switch (elementByteSize)
            {
            case 1: resolvedDataType = DML_TENSOR_DATA_TYPE_UINT8; break;
            case 2: resolvedDataType = DML_TENSOR_DATA_TYPE_UINT16; break;
            case 4: resolvedDataType = DML_TENSOR_DATA_TYPE_UINT32; break;
            case 8: resolvedDataType = DML_TENSOR_DATA_TYPE_UINT64; break;
            default: resolvedDataType = DML_TENSOR_DATA_TYPE_UNKNOWN; break;
            }
            break;

        case DML_TENSOR_DATA_TYPE_INT8:
        case DML_TENSOR_DATA_TYPE_INT16:
        case DML_TENSOR_DATA_TYPE_INT32:
            switch (elementByteSize)
            {
            case 1: resolvedDataType = DML_TENSOR_DATA_TYPE_INT8; break;
            case 2: resolvedDataType = DML_TENSOR_DATA_TYPE_INT16; break;
            case 4: resolvedDataType = DML_TENSOR_DATA_TYPE_INT32; break;
            case 8: resolvedDataType = DML_TENSOR_DATA_TYPE_INT64; break;
            default: resolvedDataType = DML_TENSOR_DATA_TYPE_UNKNOWN; break;
            }
            break;

        case DML_TENSOR_DATA_TYPE_FLOAT32:
        case DML_TENSOR_DATA_TYPE_FLOAT64:
            switch (elementByteSize)
            {
            case 2: resolvedDataType = DML_TENSOR_DATA_TYPE_FLOAT16; break;
            case 4: resolvedDataType = DML_TENSOR_DATA_TYPE_FLOAT32; break;
            case 8: resolvedDataType = DML_TENSOR_DATA_TYPE_FLOAT64; break;
            default: resolvedDataType = DML_TENSOR_DATA_TYPE_UNKNOWN; break;
            }
            break;

        default:
            assert(false); // Could not have reached here because we only set a known subset.
        }
    }

    dataType = resolvedDataType;
}

void AppendOnnxDataTypeToNumPyArray(
    DML_TENSOR_DATA_TYPE dataType,
    bool isBackwardsEndian, // Backwards endian which stores greatest bytes at lowest bytes.
    /*inout*/ std::u8string& numPyElementType
    )
{
    numPyElementType.push_back(isBackwardsEndian ? '>' : '<');

    // https://docs.python.org/2/library/array.html#module-array
    // https://numpy.org/devdocs/reference/arrays.dtypes.html
    std::u8string_view characterCode;
    switch (dataType)
    {
    //                                 Explicit sized type      Short alias
    case DML_TENSOR_DATA_TYPE_INT8:    characterCode = U8("i1") /*'b'*/; break;
    case DML_TENSOR_DATA_TYPE_UINT8:   characterCode = U8("u1") /*'B'*/; break;
    case DML_TENSOR_DATA_TYPE_INT16:   characterCode = U8("i2") /*'h'*/; break;
    case DML_TENSOR_DATA_TYPE_INT32:   characterCode = U8("i4") /*'i'*/; break;
    case DML_TENSOR_DATA_TYPE_INT64:   characterCode = U8("i8") /*'i'*/; break;
    case DML_TENSOR_DATA_TYPE_UINT16:  characterCode = U8("u2") /*'H'*/; break;
    case DML_TENSOR_DATA_TYPE_UINT32:  characterCode = U8("u4") /*'u'*/; break;
    case DML_TENSOR_DATA_TYPE_UINT64:  characterCode = U8("u8") /*'u'*/; break;
    case DML_TENSOR_DATA_TYPE_FLOAT16: characterCode = U8("f2") /*'f'*/; break;
    case DML_TENSOR_DATA_TYPE_FLOAT32: characterCode = U8("f4") /*'f'*/; break;
    case DML_TENSOR_DATA_TYPE_FLOAT64: characterCode = U8("f8") /*'d'*/; break;
    default:                           characterCode = U8("?");  assert(false);
    }
    numPyElementType.append(characterCode);
}

void SwapBytes(/*inout*/ std::span<uint8_t> arrayByteData, uint32_t elementByteSize)
{
    switch (elementByteSize)
    {
    // case 1: NOP

    case 2:
        {
            auto s16 = reinterpret_span<uint16_t>(arrayByteData);
            for (auto& u : s16)
            {
                uint32_t v = u;
                u = ((v & 0x00FF) << 8) |
                    ((v & 0xFF00) >> 8);
            }
        }
        break;

    case 4: // 32-bit
    case 8: // 64-bit
    case 16: // 128-bit
        {
            auto s32 = reinterpret_span<uint32_t>(arrayByteData);
            for (auto& u : s32)
            {
                uint32_t v = u;
                u = ((v & 0x000000FF) << 24) |
                    ((v & 0x0000FF00) << 8)  |
                    ((v & 0x00FF0000) >> 8)  |
                    ((v & 0xFF000000) >> 24);
            }

            if (elementByteSize == 8)
            {
                for (uint32_t i = 0; i < static_cast<uint32_t>(s32.size() & ~0x1); i += 2)
                {
                    std::swap(s32[i + 0], s32[i + 1]);
                }
            }
            else if (elementByteSize == 16)
            {
                for (uint32_t i = 0; i < static_cast<uint32_t>(s32.size() & ~0x3); i += 4)
                {
                    std::swap(s32[i + 0], s32[i + 3]);
                    std::swap(s32[i + 1], s32[i + 2]);
                }
            }
        }
        break;
    }
}

class PythonDictionaryLexer
{
public:
    enum class TokenType
    {
        Error,
        End,
        OpeningBrace,
        ClosingBrace,
        OpeningParenthesis,
        ClosingParenthesis,
        Identifier,
        String,
        Colon,
        Comma,
        Number,
    };

    PythonDictionaryLexer(std::span<const char8_t> text) : text_(text)
    {
    }

    PythonDictionaryLexer(std::u8string_view text) : text_(text)
    {
    }

    PythonDictionaryLexer(std::span<const std::byte> text) : text_(reinterpret_span<const char8_t>(text))
    {
    }

    bool empty()
    {
        return text_.empty();
    }

    struct ReadStruct 
    {
        std::span<const char8_t> token;
        TokenType tokenType;
    };

    ReadStruct Read()
    {
        std::span<const char8_t> token;
        TokenType tokenType = TokenType::End;

        // Skip spaces.
        for (; !text_.empty() && isspace(uint8_t(text_.front())); text_.pop_front())
        {
        }

        if (!text_.empty())
        {
            token = text_.subspan(0, 1);
            char ch = text_.consume_front();

            switch (ch)
            {
            case '{': tokenType = TokenType::OpeningBrace; break;
            case '}': tokenType = TokenType::ClosingBrace; break;
            case '(': tokenType = TokenType::OpeningParenthesis; break;
            case ')': tokenType = TokenType::ClosingParenthesis; break;
            case ':': tokenType = TokenType::Colon; break;
            case ',': tokenType = TokenType::Comma; break;
            case '\'':
            case '\"':
                {
                    tokenType = TokenType::String;
                    char leadingQuoteMark = ch;
                    ch = 0;
                    token.pop_front(); // Skip leading quote.

                    // Read until the closing quote mark.
                    for (; !text_.empty() && (ch = text_.front(), ch != leadingQuoteMark && ch != '\r' && ch != '\n'); text_.pop_front())
                    {
                    }
                    token = {token.begin(), text_.begin()};

                    if (ch == leadingQuoteMark)
                    {
                        text_.pop_front(); // Skip closing quote mark.
                    }
                    else
                    {
                        tokenType = TokenType::Error; // Unclosed string.
                    }
                }
                break;

            case '0': case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8': case '9':
                tokenType = TokenType::Number;
                for (; !text_.empty() && (ch = text_.front(), isdigit(ch) || ch == '.'); text_.pop_front())
                {
                }
                token = {token.begin(), text_.begin()};
                break;

            default:
                // Check alphanumeric identifier.
                if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'))
                {
                    tokenType = TokenType::Identifier;
                    for (; !text_.empty() && (ch = text_.front(), isalnum(ch) || ch == '.'); text_.pop_front())
                    {
                    }
                    token = {token.begin(), text_.begin()};
                }
                else
                {
                    tokenType = TokenType::Error;
                }
            }
        }

        return { token, tokenType };
    }

    std::map<std::u8string_view, std::u8string_view> ReadDictionary()
    {
        int indentLevel = 0;

        std::map<std::u8string_view, std::u8string_view> map;
        std::u8string_view currentKey;
        std::u8string_view currentValue;
        bool haveKey = false;

        auto appendCurrentKeyValue = [&]()
        {
            if (haveKey)
            {
                map.insert({currentKey, currentValue});
                currentKey = {};
                currentValue = {};
            }
            haveKey = false;
        };

        while (true)
        {
            auto [token, tokenType] = Read();

            bool extendToken = false;
            switch (tokenType)
            {
            case TokenType::Error: goto End;
            case TokenType::End: goto End;
            case TokenType::OpeningParenthesis: ++indentLevel; break;
            case TokenType::ClosingParenthesis: --indentLevel; extendToken = (indentLevel == 1); break;
            case TokenType::OpeningBrace: ++indentLevel; break;
            case TokenType::ClosingBrace: --indentLevel; extendToken = (indentLevel == 1); break;

            case TokenType::Comma:
                if (indentLevel == 1)
                {
                    appendCurrentKeyValue();
                }
                break;

            case TokenType::Colon:
                if (indentLevel == 1)
                {
                    haveKey = true;
                }
                break;

            default:
                extendToken = true;
                break;
            }

            // Glom multiple tokens together into a larger unit.
            if (indentLevel > 1 || extendToken)
            {
                auto& keyOrValue = (haveKey) ? currentValue : currentKey;
                if (keyOrValue.empty())
                {
                    keyOrValue = { token.data(), token.size() };
                }
                else
                {
                    keyOrValue = {keyOrValue.data(), size_t(token.end() - keyOrValue.data()) };
                }
            }
        }
    End:

        appendCurrentKeyValue();

        assert(indentLevel == 0);

        return map;
    }

    void ParseIntegers(/*out*/ std::vector<int32_t>& numbers)
    {
        while (true)
        {
            auto [token, tokenType] = Read();

            switch (tokenType)
            {
            case TokenType::End:
            case TokenType::Error:
                goto End;

            case TokenType::Number:
                {
                    uint32_t value = 0;
                    std::from_chars(ToChar(token.begin()), ToChar(token.end()), /*out*/ value);
                    numbers.push_back(value);
                }
                break;

            default:
                ; // Skip anything else.
            }
        }
    End:;
    }

private:
    std::span<const char8_t> text_;
};

class PythonDictionaryWriter
{
public:
    std::u8string_view GetText() const
    {
        return text_;
    }

    std::span<const std::byte> GetBytes() const
    {
        return reinterpret_span<const std::byte>(text_);
    }

    void Append(std::u8string_view text)
    {
        text_.append(text);
    }

    void WriteKeyValueUnquoted(std::u8string_view key, std::u8string_view value)
    {
        text_.append(key);
        text_.append(U8(":"));
        text_.append(value);
        text_.append(U8(", "));
    }

    void WriteKeyValue(std::u8string_view key, std::u8string_view value)
    {
        text_.push_back('\'');
        text_.append(key);
        text_.append(U8("\':\'"));
        text_.append(value);
        text_.append(U8("\', "));
    }

    void WriteIntegers(std::span<const int32_t> numbers, std::u8string_view brackets)
    {
        if (!brackets.empty())
        {
            text_.push_back(brackets.front());
        }
        for (auto n : numbers)
        {
            char buffer[11];
            auto result = std::to_chars(std::begin(buffer), std::end(buffer), n);
            text_.append(std::begin(buffer), result.ptr);
            text_.append(U8(","));
        }
        if (!brackets.empty())
        {
            text_.push_back(brackets.back());
        }
    }

private:
    std::u8string text_;
};

struct NumPyArrayHeaderV1
{
    uint8_t signature[6]; // "\x0093NUMPY"
    uint8_t majorVersion; // == 1
    uint8_t minorVersion;
    uint16_t dictionaryLength; // Confusingly instead labeled "HEADER_LEN" in the documentation.
};

struct NumPyArrayHeaderV2
{
    uint8_t signature[6]; // "\x0093NUMPY"
    uint8_t majorVersion; // == 2
    uint8_t minorVersion;
    uint32_t dictionaryLength; // Confusingly instead labeled "HEADER_LEN" in the documentation.
};

// V3 is identical to V2 except that the strings are UTF-8. For Onnx2Text, it's irrelevant
// anyway, since the encoding applies to fields we don't care about. For all the core fields
// that matter (like "shape" and "fortran_order"), they only use the lower basic ASCII.
using NumPyArrayHeaderV3 = NumPyArrayHeaderV2;

// Version 4 is unsupported. When/if NumPy specifies it some year, the tool might work
// correctly as-is assuming no breaking changes, but it feels safer to reject it than to
// silently parse the file incorrectly. Also, NumPy will likely continue to emit V2 by
// default as a minimum compatibility bar.

void ReadNpy(
    std::span<const std::byte> fileData,
    /*out*/DML_TENSOR_DATA_TYPE& dataType,
    /*out*/std::vector<int32_t>& dimensions,
    /*out*/std::vector<std::byte>& arrayByteData
    )
{
    dataType = DML_TENSOR_DATA_TYPE_UNKNOWN;
    dimensions.clear();
    arrayByteData.clear();

    using namespace std::literals;

    if (fileData.size_bytes() < sizeof(NumPyArrayHeaderV1))
    {
        throw std::ios::failure("NumPy array header signature is invalid.");
    }

    auto& headerV1 = read_as<NumPyArrayHeaderV1>(fileData);
    auto& headerV2 = read_as<NumPyArrayHeaderV2>(fileData);
    if (headerV1.majorVersion >= 4)
    {
        throw std::ios::failure("Versions >= 4 unsupported.");
    }

    size_t dictionaryOffset = (headerV1.majorVersion >= 2) ? sizeof(NumPyArrayHeaderV2) : sizeof(NumPyArrayHeaderV1);
    size_t dictionaryLength = (headerV1.majorVersion >= 2) ? headerV2.dictionaryLength : headerV1.dictionaryLength;
    size_t dataByteOffset = dictionaryOffset + dictionaryLength;

    PythonDictionaryLexer lexer(fileData.subrange(dictionaryOffset, dataByteOffset));
    std::map<std::u8string_view, std::u8string_view> dictionary = lexer.ReadDictionary();

    bool isBackwardsEndian = false;
    bool hasIncreasingStrides = false;

    for (auto& i : dictionary)
    {
        if (i.first == std::u8string_view(U8("descr")))
        {
            MapNumPyArrayDataTypeToDml(i.second, /*out*/ dataType, /*out*/ isBackwardsEndian);
        }
        else if (i.first == std::u8string_view(U8("fortran_order")))
        {
            hasIncreasingStrides = (i.second == std::u8string_view(U8("True")));
        }
        else if (i.first == std::u8string_view(U8("shape")))
        {
            PythonDictionaryLexer shapeLexer(i.second);
            shapeLexer.ParseIntegers(dimensions);
        }
    }

    arrayByteData.assign(fileData.data() + dataByteOffset, fileData.end());
    const uint32_t elementByteSize = GetByteSizeFromDataType(dataType);
    const uint32_t totalElementCount = ComputeElementCount(dimensions);
    const uint32_t totalByteSize = elementByteSize * totalElementCount;
    if (arrayByteData.size() < totalByteSize)
    {
        arrayByteData.resize(totalByteSize);
    }

    // Assuming that we're running on a logical endian machine.
    // If not, lots of other places would break too anyway.
    if (isBackwardsEndian)
    {
        SwapBytes(/*inout*/ reinterpret_span<uint8_t>(arrayByteData), totalByteSize);
    }
    if (hasIncreasingStrides)
    {
        throw std::ios::failure("Fortran stride order unsupported.");
    }
}

// Writes tensor data to in memory file data (not directly to file).
void WriteNpy(
    std::span<const std::byte> arrayByteData,
    DML_TENSOR_DATA_TYPE dataType,
    std::span<const int32_t> dimensions,
    /*out*/std::vector<std::byte>& fileData
    )
{
    NumPyArrayHeaderV1 headerFixedPart = { {uint8_t('\x0093'),'N','U','M','P','Y'}, 1,0, 0 };

    PythonDictionaryWriter dictionaryWriter;
    PythonDictionaryWriter numberWriter;

    // Format dictionary fields.
    std::u8string numPyElementType;
    AppendOnnxDataTypeToNumPyArray(dataType, /*isBackwardsEndian*/ false, /*inout*/ numPyElementType);
    numberWriter.WriteIntegers(dimensions, U8("()"));

    dictionaryWriter.Append(U8("{"));
    dictionaryWriter.WriteKeyValue(U8("descr"), numPyElementType);
    dictionaryWriter.WriteKeyValueUnquoted(U8("'fortran_order'"), U8("False"));
    dictionaryWriter.WriteKeyValueUnquoted(U8("'shape'"), numberWriter.GetText());
    dictionaryWriter.Append(U8("}"));

    // Compute header length for alignment.
    uint32_t headerLength = sizeof(headerFixedPart);
    headerLength += static_cast<uint32_t>(dictionaryWriter.GetText().size());
    headerLength++; // For new line.
    headerLength = (headerLength + 63) & ~63; // For rounding up to multiple of 64 alignment.

    // Write header, including fixed size part, dictionary, and alignment padding.
    headerFixedPart.dictionaryLength = static_cast<uint16_t>(headerLength - sizeof(headerFixedPart));
    append_data(/*inout*/ fileData, { reinterpret_cast<const std::byte*>(&headerFixedPart), sizeof(headerFixedPart) });
    append_data(/*inout*/ fileData, dictionaryWriter.GetBytes());
    fileData.insert(fileData.end(), headerLength - fileData.size(), std::byte{' '});
    fileData.back() = std::byte{ '\x000A' }; // Terminate with new line.
    // Note the spec says "It is terminated by a newline (\n) and padded with spaces (\x20)",
    // but that's wrong. It's actually "padding with spaces and then terminated by a newline".
    // Otherwise Numpy 1.18.5 barfs (1.19 works fine either way).
    // https://numpy.org/devdocs/reference/generated/numpy.lib.format.html

    append_data(/*inout*/ fileData, arrayByteData);
}
