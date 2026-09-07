#pragma once

#include <string>
#include <string_view>

namespace kfbim::app3d {

// Encode one CSV field according to RFC 4180.  Fields containing a comma,
// quote, CR, or LF are enclosed in double quotes and embedded quotes are
// doubled.  Keeping this helper independent of the large 3D driver makes the
// output contract directly unit-testable.
inline std::string rfc4180_csv_field(std::string_view value)
{
    const bool needs_quotes = value.find_first_of(",\"\r\n")
        != std::string_view::npos;
    if (!needs_quotes)
        return std::string(value);

    std::string encoded;
    encoded.reserve(value.size() + 2);
    encoded.push_back('"');
    for (const char character : value) {
        if (character == '"')
            encoded.push_back('"');
        encoded.push_back(character);
    }
    encoded.push_back('"');
    return encoded;
}

} // namespace kfbim::app3d
