#include "scidf.h"
#include "spade.h"

#include "solver.h"
#include "impulse.h"
#include "sample.h"
#include "image.h"

using coor_t = double;
using real_t = float;

inline std::string getfname(const std::size_t& i)
{
    std::string output = std::to_string(i);
    while (output.length() < 8) output = "0" + output;
    return "snap_" + output;
}

void entry(spade::parallel::pool_t& pool)
{
    scidf::node_t input;
    scidf::read("input.sdf", input);
    sl::solver slv(input, pool);
    std::string out_dir_str = input["out_dir"];
    std::filesystem::path out_dir(out_dir_str);
    spade::io::mkdir(out_dir);
    spade::io::mkdir(out_dir / "misc");
    spade::io::mkdir(out_dir / "hist");
    spade::io::mkdir(out_dir / "img");
    spade::io::output_vtk(out_dir / "misc" / "grid.vtk", slv.get_grid());
    
    real_t d0_default = input["d0"];
    real_t d1_default = input["d1"];
    real_t d2_default = input["d2"];
    
    // spade::algs::fill_array(slv.get_speed(), [=] _sp_hybrid (const spade::coords::point_t<double>& x){
    //     auto xhat = x[0] - 0.5;
    //     auto yhat = x[1] - 0.5;
    //     auto r = sqrt(xhat*xhat + yhat*yhat);
    //     auto back = 1 + sin(30*x[0])*cos(20*x[1]);
    //     return 1 - 0.8*exp(-13*r*r) + 0.6*back;
    // });
    
    spade::algs::fill_array(slv.get_speed(), [=] _sp_hybrid (const spade::coords::point_t<double>& x){
        int n = 5;
        float wid = 0.03;
        float dx = 1.0/n;
        int px = 1;
        int py = 1;
        
        if (x[0] > px*dx && x[0] < (px+1)*dx && x[1] > py*dx && x[1] < (py+1)*dx) return 1;
        int ix = floor(x[0]/dx);
        int iy = floor(x[1]/dx);
        
        float cx = (float(ix) + 0.5)*dx;
        float cy = (float(iy) + 0.5)*dx;
        
        float xmin = cx - (0.5*dx - 0.5*wid);
        float xmax = cx + (0.5*dx - 0.5*wid);
        float ymin = cy - (0.5*dx - 0.5*wid);
        float ymax = cy + (0.5*dx - 0.5*wid);
        if (x[0] < xmax && x[0] > xmin && x[1] < ymax && x[1] > ymin) return 0;
        
        xmin = 0.45;
        xmax = 0.55;
        ymin = 0.45;
        ymax = 0.55;
        if (x[0] < xmax && x[0] > xmin && x[1] < ymax && x[1] > ymin) return 0;
        return 1;
    });
    
    spade::algs::fill_array(slv.get_d0(), [=] _sp_hybrid (const spade::coords::point_t<double>& x){
        return d0_default;
    });
    
    spade::algs::fill_array(slv.get_d1(), [=] _sp_hybrid (const spade::coords::point_t<double>& x){
        return d1_default;
    });
    
    spade::algs::fill_array(slv.get_d2(), [=] _sp_hybrid (const spade::coords::point_t<double>& x){
        return d2_default;
    });
    
    spade::io::output_vtk(out_dir / "misc", "spd", slv.get_speed());
    
    print("timestep:", slv.get_dt());
    
    // fill impulses
    auto impulses = sl::read_impulses(input["Impulses"]);
    print("num. impulses:", impulses.size());
    
    auto [names, samples] = sl::read_samples(input["Samples"], slv);
    print("Sample names:");
    for (const auto& n: names)
    {
        print(" ", n);
    }
    
    sl::sample_stream samples_out(out_dir / "samples.dat", names);
    
    slv.add_init_condition(impulses);
    
    spade::io::mkdir(out_dir / "surf");
    spade::io::output_vtk(out_dir / "surf", "init", slv.solution());
    std::size_t max_step = input["max_step"];
    std::size_t out_itvl = input["out_itvl"];
    std::size_t spl_itvl = input["spl_itvl"];
    
    std::size_t img_itvl = input["img_itvl"];
    float img_min = input["img_min"];
    float img_max = input["img_max"];
    int   img_wid = input["img_wid"];
    int   img_hei = input["img_hei"];
    
    for (std::size_t i = 0; i <= max_step; ++i)
    {
        if (i % out_itvl == 0) spade::io::output_vtk(out_dir / "hist", getfname(i), slv.solution());
        if (i % spl_itvl == 0) sl::write_samples(samples_out, samples, slv.solution());
        if (i % img_itvl == 0) sl::write_image(std::string(out_dir / "img" / getfname(i)) + ".png", slv.solution(), img_min, img_max, img_wid, img_hei);
        print(i, slv.get_time());
        slv.advance();
    }
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
