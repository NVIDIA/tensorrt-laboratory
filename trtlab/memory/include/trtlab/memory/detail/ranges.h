/* Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include <numeric>
#include <utility>
#include <vector>
#include <type_traits>

namespace trtlab
{
    namespace memory
    {
        namespace detail
        {
            template <typename T>
            std::vector<std::pair<T, T>> find_ranges(const std::vector<T>& values)
            {
                static_assert(std::is_integral<T>::value, "only integral types allowed");

                auto copy = values;
                sort(copy.begin(), copy.end());

                std::vector<std::pair<T, T>> ranges;

                auto it  = copy.cbegin();
                auto end = copy.cend();

                while (it != end)
                {
                    auto low  = *it;
                    auto high = *it;
                    for (T i = 0; it != end && low + i == *it; it++, i++)
                    {
                        high = *it;
                    }
                    ranges.push_back(std::make_pair(low, high));
                }

                return ranges;
            }

            template <typename T>
            std::string print_ranges(const std::vector<std::pair<T, T>>& ranges)
            {
                return std::accumulate(std::begin(ranges), std::end(ranges), std::string(), [](std::string r, std::pair<T, T> p) {
                    if (p.first == p.second)
                    {
                        return r + (r.empty() ? "" : ",") + std::to_string(p.first);
                    }
                    else
                    {
                        return r + (r.empty() ? "" : ",") + std::to_string(p.first) + "-" + std::to_string(p.second);
                    }
                });
            }

        } // namespace detail
    }     // namespace memory
} // namespace trtlab