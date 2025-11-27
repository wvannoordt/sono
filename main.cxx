#include "scidf.h"
#include "spade.h"

#include "solver.h"

using coor_t = double;
using real_t = float;

void entry(spade::parallel::pool_t& pool)
{
    scidf::node_t input;
    scidf::read("input.sdf", input);
    sl::solver slv(input, pool);
    std::string out_dir_str = input["out_dir"];
    std::filesystem::path out_dir(out_dir_str);
    spade::io::mkdir(out_dir);
    spade::io::output_vtk(out_dir / "grid.vtk", slv.get_grid());
}

int main(int argc, char** argv)
{
    spade::parallel::compute_env_t env(&argc, &argv, "0");
    env.exec([&](spade::parallel::pool_t& pool)
    {
        entry(pool);
    });
    return 0;
}
