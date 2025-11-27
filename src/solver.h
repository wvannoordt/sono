#include "scidf.h"
#include "spade.h"

namespace sl
{
    class solver
    {
        public:
        using coords_type = double;
        using value_type  = float;
        using blocks_type = spade::amr::amr_blocks_t<coords_type, spade::ctrs::array<int, 2>>;
        using grid_type   = spade::grid::cartesian_grid_t<spade::ctrs::array<int, 2>, spade::coords::identity<coords_type>, blocks_type, spade::parallel::pool_t>;
        using array_type  = spade::grid::grid_array<grid_type, value_type, spade::device::best_type, spade::mem_map::ttiled_small_t, spade::grid::cell_centered>;
        
        private:
        std::unique_ptr<grid_type> grid;
        
        public:
        solver(scidf::node_t& input, spade::parallel::pool_t& pool)
        {
            spade::coords::identity<coords_type> coords;
            spade::ctrs::array<int, 2> num_blocks = input["num_blocks"];
            spade::ctrs::array<int, 2> num_cells  = input["num_cells"];
            spade::ctrs::array<coords_type, 4> bnds  = input["bounds"];
            spade::bound_box_t<coords_type, 2> bounds{bnds[0], bnds[1], bnds[2], bnds[3]};
            spade::amr::amr_blocks_t blocks(num_blocks, bounds);
            spade::grid::cartesian_grid_t grid_temp(num_cells, blocks, coords, spade::ctrs::make_array(false, false), pool);
            
            grid = std::make_unique<grid_type>(grid_temp);
            
            // real_t q = 0.0;
            // spade::ctrs::array<int, 2> num_exch{2, 2};
            // spade::grid::grid_array disp(grid, q, num_exch, spade::device::best, spade::mem_map::tiled_small);
        }
        
        const auto& get_grid() const { return *grid; }
    };
}