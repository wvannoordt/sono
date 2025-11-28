#pragma once

#include "scidf.h"
#include "spade.h"
#include "impulse.h"

namespace sl
{
    template <typename rtype>
	struct wstate
	: public spade::ctrs::arithmetic_array_t<rtype, 2, wstate<rtype>>
    {
        using base_t = spade::ctrs::arithmetic_array_t<rtype, 2, wstate<rtype>>;
        using base_t::base_t;
        _sp_hybrid wstate(){}
        _sp_hybrid rtype& u() {return (*this)[0];}
        _sp_hybrid rtype& v() {return (*this)[1];}
        _sp_hybrid const rtype& u() const {return (*this)[0];}
        _sp_hybrid const rtype& v() const {return (*this)[1];}
        static std::string name(uint idx)
        {
            spade::ctrs::array<std::string, 2> names("disp", "vel");
            return names[idx];
        }
    };
    
    class solver
    {
        public:
        using coords_type = double;
        using value_type  = float;
        using state_type  = wstate<value_type>;
        using blocks_type = spade::amr::amr_blocks_t<coords_type, spade::ctrs::array<int, 2>>;
        using grid_type   = spade::grid::cartesian_grid_t<spade::ctrs::array<int, 2>, spade::coords::identity<coords_type>, blocks_type, spade::parallel::pool_t>;
        using array_type  = spade::grid::grid_array<grid_type, state_type, spade::device::best_type, spade::mem_map::tlinear_t, spade::grid::cell_centered>;
        
        private:
        std::unique_ptr<grid_type> grid;
        std::unique_ptr<array_type> sol;
        
        double cfl;
        double spd;
        
        double dt;
        
        public:
        solver(scidf::node_t& input, spade::parallel::pool_t& pool)
        {
            spade::coords::identity<coords_type> coords;
            spade::ctrs::array<int, 2> num_blocks = input["num_blocks"];
            spade::ctrs::array<int, 2> num_cells  = input["num_cells"];
            spade::ctrs::array<coords_type, 4> bnds  = input["bounds"];
            
            
            cfl = double(input["cfl"]);
            spd = double(input["spd"]);
            
            spade::bound_box_t<coords_type, 2> bounds{bnds[0], bnds[1], bnds[2], bnds[3]};
            spade::amr::amr_blocks_t blocks(num_blocks, bounds);
            spade::grid::cartesian_grid_t grid_temp(num_cells, blocks, coords, spade::ctrs::make_array(false, false), pool);
            grid = std::make_unique<grid_type>(std::move(grid_temp));
            
            spade::ctrs::array<int, 2> num_exch{2, 2};
            sol = std::make_unique<array_type>(*grid, num_exch);
            *sol = 0;
            
            this->set_dt();
        }
        
        const auto& get_grid() const { return *grid; }
        double get_dt() const { return dt; }
        auto& solution() { return *sol; }
        const auto& solution() const { return *sol; }
        
        void add_init_condition(const std::vector<impulse>& imps)
        {
            for (const auto& imp: imps)
            {
                spade::algs::fill_array(this->solution(), [=] _sp_hybrid (const state_type& w, const spade::coords::point_t<coords_type>& x){
                    auto output = w;
                    spade::ctrs::array<coords_type, 2> xx(x.x(), x.y()); // annoying type strictness
                    output.u() += imp.eval(xx);
                    return output;
                });
            }
        }
        
        private:
        void set_dt()
        {
            const auto dxmin = spade::ctrs::array_norm(grid->compute_dx_min());
            dt = cfl * dxmin / spd;
        }
    };
}