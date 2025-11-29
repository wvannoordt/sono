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
        using coords_type   = double;
        using value_type    = float;
        using state_type    = wstate<value_type>;
        using blocks_type   = spade::amr::amr_blocks_t<coords_type, spade::ctrs::array<int, 2>>;
        using grid_type     = spade::grid::cartesian_grid_t<spade::ctrs::array<int, 2>, spade::coords::identity<coords_type>, blocks_type, spade::parallel::pool_t>;
        using array_type    = spade::grid::grid_array<grid_type, state_type, spade::device::best_type, spade::mem_map::tlinear_t, spade::grid::cell_centered>;
        using scalr_type    = spade::grid::grid_array<grid_type, value_type, spade::device::best_type, spade::mem_map::tlinear_t, spade::grid::cell_centered>;
        using exchange_type = spade::grid::arr_exchange_t<array_type>;
        
        private:
        std::unique_ptr<grid_type> grid;
        std::unique_ptr<array_type> sol;
        std::unique_ptr<array_type> sol2;
        std::unique_ptr<scalr_type> spd;
        std::unique_ptr<array_type> rhs;
        
        
        exchange_type ex;
        
        double cfl;
        
        double dt;
        double time;
        
        public:
        solver(scidf::node_t& input, spade::parallel::pool_t& pool)
        {
            spade::coords::identity<coords_type> coords;
            spade::ctrs::array<int, 2> num_blocks = input["num_blocks"];
            spade::ctrs::array<int, 2> num_cells  = input["num_cells"];
            spade::ctrs::array<coords_type, 4> bnds  = input["bounds"];
            
            
            cfl = double(input["cfl"]);
            time = 0;
            spade::bound_box_t<coords_type, 2> bounds{bnds[0], bnds[1], bnds[2], bnds[3]};
            spade::amr::amr_blocks_t blocks(num_blocks, bounds);
            spade::grid::cartesian_grid_t grid_temp(num_cells, blocks, coords, spade::ctrs::make_array(false, false), pool);
            grid = std::make_unique<grid_type>(std::move(grid_temp));
            
            spade::ctrs::array<int, 2> num_exch{2, 2};
            spade::ctrs::array<int, 2> n0{0, 0};
            sol   = std::make_unique<array_type>(*grid, num_exch);
            sol2  = std::make_unique<array_type>(*grid, num_exch);
            rhs   = std::make_unique<array_type>(*grid, num_exch);
            spd   = std::make_unique<scalr_type>(*grid, num_exch);
            *sol  = 0;
            *sol2 = 0;
            *rhs  = 0;
            *spd  = 1;
            ex = spade::grid::make_exchange(*sol, 2);
            this->set_dt();
            ex.exchange(*sol, sol->get_grid().group());
            *sol2 = *sol;
        }
        
        const auto& get_grid() const { return *grid; }
        double get_dt() const { return dt; }
        auto& solution() { return *sol; }
        const auto& solution() const { return *sol; }
        auto get_time() const { return time; }
        auto& get_speed() { return *spd; }
        const auto& get_speed() const { return *spd; }
        
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
        
        void advance()
        {
            auto dt_loc = value_type(dt);
        
            for (int stage = 0; stage < 3; ++stage)
            {
                const auto grid_img  = grid->image(sol->device());
                const auto q_old_img = sol->image();   // stage input
                auto       q_new_img = sol2->image();  // stage output
                auto       aux_img   = rhs->image();   // holds q^n, then final RHS
                
                auto t_loc  = value_type(time);
                if (stage == 1) t_loc = value_type(time + dt);
                if (stage == 2) t_loc = value_type(time + 0.5*dt);
                
                const auto c_arr_img = spd->image();
        
                spade::algs::for_each(*rhs, [=] _sp_hybrid (const spade::grid::cell_idx_t& icell) mutable
                {
                    // Metric / spacing
                    const auto inv_dx_native = grid_img.get_inv_dx(icell.lb());
                    constexpr int size = std::decay_t<decltype(inv_dx_native)>::size();
                    spade::ctrs::array<value_type, size> inv_dx;
                    #pragma unroll
                    for (int d = 0; d < size; ++d) inv_dx[d] = inv_dx_native[d];
        
                    // Current stage state q (READ-ONLY field)
                    const auto q = q_old_img.get_elem(icell);
        
                    // Compute L(q) into rhs_loc
                    state_type rhs_loc = 0;
                    rhs_loc.u() = q.v();
                    rhs_loc.v() = value_type(0);
                    const auto c_loc = c_arr_img.get_elem(icell);
                    for (int d = 0; d < 2; ++d)
                    {
                        auto inv_ddx2 = inv_dx[d]*inv_dx[d];
        
                        auto i   = icell;
                        auto im  = i;  im.i(d) --;
                        auto imm = im; imm.i(d)--;
                        auto ip  = i;  ip.i(d) ++;
                        auto ipp = ip; ipp.i(d)++;
        
                        const auto qc  = q_old_img.get_elem(i);
                        const auto qm  = q_old_img.get_elem(im);
                        const auto qmm = q_old_img.get_elem(imm);
                        const auto qp  = q_old_img.get_elem(ip);
                        const auto qpp = q_old_img.get_elem(ipp);
        
                        // const auto ddx = (qp.u() - value_type(2.0)*q.u() + qm.u()) * inv_ddx2;
                        
                        const auto ddx =
                                ( - qpp.u()
                                  + value_type(16.0) * qp.u()
                                  - value_type(30.0) * qc.u()
                                  + value_type(16.0) * qm.u()
                                  - qmm.u()
                                ) * (inv_ddx2 / value_type(12.0));
                        rhs_loc.v() += c_loc*c_loc*ddx;
                    }
        
                    if (stage == 0)
                    {
                        // ---- Stage 1: u^(1) = u^n + dt * L(u^n)
                        // Store q^n in aux_img in this pass.
                        aux_img.set_elem(icell, q);           // q^n
                        auto q_new = q + dt_loc * rhs_loc;    // u^(1)
                        q_new_img.set_elem(icell, q_new);
                    }
                    else if (stage == 1)
                    {
                        // ---- Stage 2:
                        // u^(2) = 3/4 u^n + 1/4 (u^(1) + dt * L(u^(1)))
                        const auto qn  = aux_img.get_elem(icell);  // q^n
                        auto tmp       = q + dt_loc * rhs_loc;     // u^(1) + dt*k1
                        auto q_new     = value_type(3.0/4.0)*qn
                                       + value_type(1.0/4.0)*tmp;  // u^(2)
                        q_new_img.set_elem(icell, q_new);
                        // aux_img stays as q^n
                    }
                    else // stage == 2
                    {
                        // ---- Stage 3:
                        // u^{n+1} = 1/3 u^n + 2/3 (u^(2) + dt * L(u^(2)))
                        const auto qn  = aux_img.get_elem(icell);  // q^n
                        auto tmp       = q + dt_loc * rhs_loc;     // u^(2) + dt*k2
                        auto q_new     = value_type(1.0/3.0)*qn
                                       + value_type(2.0/3.0)*tmp;  // u^{n+1}
                        q_new_img.set_elem(icell, q_new);
        
                        // Final RK RHS: (u^{n+1} - u^n)/dt
                        auto rhs_final = (q_new - qn) / dt_loc;
                        aux_img.set_elem(icell, rhs_final);
                    }
                });
        
                // Swap so that 'sol' is always the current stage solution
                std::swap(sol, sol2);
        
                // Exchange halos for the field that will be READ next
                ex.exchange(*sol, sol->get_grid().group());
            }
        
            // After the loop:
            //   sol  -> q^{n+1}
            //   rhs  -> (q^{n+1} - q^n)/dt   (effective RK RHS)
            //   sol2 -> u^(2) (not needed)
            time += dt;
        }
        
        private:
        void set_dt()
        {
            const auto dxmin = spade::ctrs::array_norm(grid->compute_dx_min());
            dt = cfl * dxmin / 1.0;
        }
    };
}