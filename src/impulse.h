#pragma once

#include "scidf.h"
#include "spade.h"

namespace sl
{
    struct impulse
    {
        using coords_type = double;
        using value_type  = float;
        
        spade::ctrs::array<coords_type, 2> x;
        value_type ampl, radius;
        
        // Smoothstep helper: S(s) = 6s^5 - 15s^4 + 10s^3, C2 on [0,1]
        _sp_hybrid static value_type smootherstep(value_type s)
        {
            // Clamp for safety
            if (s <= value_type(0)) return value_type(0);
            if (s >= value_type(1)) return value_type(1);
    
            const value_type s2 = s * s;
            const value_type s3 = s2 * s;
            // S(s) = s^3 (6 s^2 - 15 s + 10)
            return s3 * (value_type(6) * s2 - value_type(15) * s + value_type(10));
        }
    
        _sp_hybrid value_type eval(const spade::ctrs::array<coords_type, 2>& x_in) const
        {
            // Cast coordinate differences into value_type
            const value_type dx0 = static_cast<value_type>(x_in[0] - x[0]);
            const value_type dx1 = static_cast<value_type>(x_in[1] - x[1]);
    
            const value_type r = static_cast<value_type>(
                sqrt(dx0 * dx0 + dx1 * dx1)
            );
    
            const value_type R = radius;
            if (R <= value_type(0)) return value_type(0);
    
            // Decay over 0.1 * radius
            constexpr value_type falloff_ratio = value_type(0.1);
            const value_type delta = falloff_ratio * R;
            const value_type R2    = R + delta;
    
            // Inside core: constant amplitude
            if (r <= R)  return ampl;
    
            // Outside support: zero
            if (r >= R2) return value_type(0);
    
            // Transition region: smooth C2 decay from 1 -> 0
            const value_type s = (r - R) / delta;          // in (0,1)
            const value_type S = smootherstep(s);          // 0 -> 1
            const value_type w = value_type(1) - S;        // 1 -> 0
    
            return ampl * w;
        }
    };
    
    inline std::vector<impulse> read_impulses(scidf::node_t& input)
    {
        std::vector<impulse> out;
        for (auto& [_, sec] : input.children)
        {
            impulse i;
            i.x      = spade::ctrs::array<impulse::coords_type, 2>(sec["location"]);
            i.ampl   = impulse::value_type(sec["ampl"]);
            i.radius = impulse::value_type(sec["radius"]);
            out.emplace_back(i);
        }
        return out;
    }
}