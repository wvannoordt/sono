#pragma once

#include "scidf.h"
#include "spade.h"
#include "solver.h"

namespace sl
{
    struct impulse
    {
        using coords_type = double;
        using value_type  = float;
        
        spade::ctrs::array<coords_type, 2> x;
        value_type ampl, fwhm;
    };
    
    inline std::vector<impulse> read_impulses(scidf::node_t& input)
    {
        std::vector<impulse> out;
        for (auto& [_, sec] : input.children)
        {
            impulse i;
            i.x = spade::ctrs::array<impulse::coords_type, 2>(input["location"]);
            i.ampl = impulse::value_type(input["ampl"]);
            i.fwhm = impulse::value_type(input["fwhm"]);
        }
        return out;
    }
}