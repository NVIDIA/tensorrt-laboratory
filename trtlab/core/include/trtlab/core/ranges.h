#pragma once
#include <numeric>
#include <vector>
#include <utility>
#include <type_traits>

namespace trtlab
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

} // namespace trtlab