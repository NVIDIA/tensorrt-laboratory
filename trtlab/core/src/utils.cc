/* Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "trtlab/core/utils.h"

#include <math.h>
#include <stdio.h>

#include <cmath>
#include <regex>

#include <glog/logging.h>

namespace trtlab {
/**
 * @brief Converts bytes into a more friend human readable format
 *
 * @param bytes
 * @return std::string
 */
std::string BytesToString(size_t bytes)
{
    // C++ implentation inspired from: https://stackoverflow.com/questions/3758606
    char buffer[50];
    int unit = 1024;
    const char prefixes[] = "KMGTPE";
    if(bytes < unit)
    {
        sprintf(buffer, "%ld B", bytes);
        return std::string(buffer);
    }
    int exp = (int)(std::log(bytes) / std::log(unit));
    sprintf(buffer, "%.1f %ciB", bytes / std::pow(unit, exp), prefixes[exp - 1]);
    return std::string(buffer);
}

std::uint64_t StringToBytes(const std::string str)
{
    // https://regex101.com/r/UVm5wT/1
    std::smatch m;
    std::regex r("(\\d+[.\\d+]*)([KMGTkmgt]*)([i]*)[bB]");
    std::map<char, int> prefix = {
        {'k', 1}, {'m', 2}, {'g', 3}, {'t', 4}, {'K', 1}, {'M', 2}, {'G', 3}, {'T', 4},
    };

    if(!std::regex_search(str, m, r))
        LOG(FATAL) << "Unable to convert \"" << str << "\" to bytes. "
                   << "Expected format: 10b, 1024B, 1KiB, 10MB, 2.4gb, etc.";

    const std::uint64_t base = m.empty() || (m.size() > 3 && m[3] == "") ? 1000 : 1024;
    auto exponent = prefix[m[2].str()[0]];
    auto scalar = std::stod(m[1]);
    return (std::uint64_t)(scalar * std::pow(base, exponent));
}

} // namespace trtlab