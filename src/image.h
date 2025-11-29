#pragma once

#include "solver.h"
#include "spade.h"
#include "extensions/gfx/gfx.h"

namespace sl
{
    using rgb_type    = spade::ctrs::array<float, 3>;
    
    struct color_map_t {
        constexpr static int max_colors = 9;
        spade::ctrs::array<rgb_type, max_colors> colors{};
        int num_colors = 0;
    
        _sp_hybrid bool add_color(const rgb_type& c) {
            if (num_colors >= (int)colors.size()) return false;
            colors[num_colors++] = c;
            return true;
        }
    
        _sp_hybrid static float clamp01(float x) {
            return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
        }
    
        _sp_hybrid static float lerp(float a, float b, float t) {
            return a + (b - a) * t;
        }
    
        _sp_hybrid static float normalize_t(float t, float tmin, float tmax) {
            if (tmin == tmax) return 0.5f;
            if (tmin > tmax) { float tmp = tmin; tmin = tmax; tmax = tmp; }
            return clamp01((t - tmin) / (tmax - tmin));
        }
    
        // Core interpolation between color stops
        _sp_hybrid rgb_type sample_linear(float t) const {
            if (num_colors == 0) return rgb_type();
            if (num_colors == 1) return colors[0];
    
            t = clamp01(t);
            float pos = t * (float)(num_colors - 1);
            int i0 = (int)floorf(pos);
            int i1 = (i0 >= num_colors - 1) ? (num_colors - 1) : (i0 + 1);
            float u = pos - (float)i0;
    
            const rgb_type& c0 = colors[i0];
            const rgb_type& c1 = colors[i1];
            return rgb_type(
                lerp(c0[0], c1[0], u),
                lerp(c0[1], c1[1], u),
                lerp(c0[2], c1[2], u)
            );
        }
    
        // Public sampling API
        template <class T1, class T2, class T3>
        _sp_hybrid rgb_type get_color(T1 t, T2 tmin, T3 tmax) const {
            float u = normalize_t((float)t, (float)tmin, (float)tmax);
            return sample_linear(u);
        }
    
        // Convenience overloads
        _sp_hybrid rgb_type get_color() const {
            return (num_colors > 0) ? colors[0] : rgb_type();
        }
    
        _sp_hybrid rgb_type get_color(int idx) const {
            if (num_colors <= 0) return rgb_type();
            if (idx < 0) idx = 0;
            if (idx >= num_colors) idx = num_colors - 1;
            return colors[idx];
        }
    };
    
    inline void write_image(const std::string& fname, const typename solver::array_type& arr, float vmin, float vmax, int wid, int hei)
    {
        color_map_t cmap;
        constexpr float inv = 1.0f / 255.0f;
        cmap.add_color(rgb_type( 21*inv,  27*inv, 148*inv));
        cmap.add_color(rgb_type(  0*inv,  43*inv, 255*inv));
        cmap.add_color(rgb_type(146*inv, 255*inv, 255*inv));
        cmap.add_color(rgb_type(255*inv, 255*inv, 255*inv));
        cmap.add_color(rgb_type(255*inv, 190*inv,   0*inv));
        cmap.add_color(rgb_type(255*inv,   0*inv,   0*inv));
        cmap.add_color(rgb_type( 94*inv,   0*inv,   0*inv));
        
        using state_type = typename solver::array_type::alias_type;
        const auto devc = arr.device();
        using devc_t = std::decay_t<decltype(devc)>;
        spade::ext::gfx::image_buffer_t<float, devc_t> r;
        spade::ext::gfx::image_buffer_t<float, devc_t> g;
        spade::ext::gfx::image_buffer_t<float, devc_t> b;
        r.set_size(hei, wid, 0.0f);
        g.set_size(hei, wid, 0.0f);
        b.set_size(hei, wid, 0.0f);
        
        const auto bv = arr.get_grid().compute_bvh(spade::partition::local);
        const auto bvh_img = bv.image(devc);
        const auto grid_img = arr.get_grid().image(spade::partition::local, devc);
        const auto num_blk = arr.get_grid().get_num_local_blocks();
        const auto arr_img = arr.image();
        
        auto rimg = r.image();
        auto gimg = g.image();
        auto bimg = b.image();
        auto glob_bnds = arr.get_grid().get_bounds();
        using prec_type = float;
        
        spade::ext::gfx::fill_buffer(r, [=] _sp_hybrid (const int row, const int col) mutable
        {
            float r_out = 0;
            float g_out = 0;
            float b_out = 0;
            
            const auto dst = [&](const auto& xx, const std::size_t& ilb)
            {
                const auto bbx = grid_img.get_bounding_box(ilb);
                return bvh_img.box_min_dist(bbx, xx);
            };
            
            auto pix_x = glob_bnds.min(0) + (double(col)/double(wid - 1))*glob_bnds.size(0);
            auto pix_y = glob_bnds.min(1) + (1.0 - (double(row)/double(hei - 1)))*glob_bnds.size(1);
            typename std::decay_t<decltype(bv)>::pnt_t xx(pix_x, pix_y, 0);
            std::size_t lb = bvh_img.find_closest_element(xx, dst);
            // printf("%d\n", int(lb));
            
            const auto& bnd = grid_img.get_bounding_box(lb);
            const auto& dx  = grid_img.get_dx(lb);
    
            // Sample position
            const double x = pix_x;
            const double y = pix_y;
    
            // Block physical extents
            const double xmin = bnd.min(0);
            const double xmax = bnd.max(0);
            const double ymin = bnd.min(1);
            const double ymax = bnd.max(1);
    
            // Approximate number of cells in this block in i,j
            const prec_type nx_d = (xmax - xmin) / dx[0];
            const prec_type ny_d = (ymax - ymin) / dx[1];
            const int    nx   = (nx_d > 0.0) ? static_cast<int>(nx_d + 0.5) : 1;
            const int    ny   = (ny_d > 0.0) ? static_cast<int>(ny_d + 0.5) : 1;
    
            // Logical coordinates in cell index space
            prec_type xi = (x - xmin) / dx[0];
            prec_type yi = (y - ymin) / dx[1];
    
            // Clamp inside valid range
            if (xi < 0.0) xi = 0.0;
            if (yi < 0.0) yi = 0.0;
            if (nx > 1 && xi > nx - 1) xi = nx - 1;
            if (ny > 1 && yi > ny - 1) yi = ny - 1;
    
            // Lower/upper indices and interpolation weights
            const int i0 = static_cast<int>(xi);  // floor, since xi >= 0
            const int j0 = static_cast<int>(yi);
            const int i1 = (nx > 1) ? ((i0 + 1 < nx) ? i0 + 1 : i0) : i0;
            const int j1 = (ny > 1) ? ((j0 + 1 < ny) ? j0 + 1 : j0) : j0;
    
            const prec_type tx = xi - static_cast<prec_type>(i0);
            const prec_type ty = yi - static_cast<prec_type>(j0);
    
            // For now assume a single layer in k (2D case) => kk = 0
            const int kk = 0;
    
            // Fetch the four surrounding cell values
            spade::grid::cell_idx_t c00(i0, j0, kk, lb);
            spade::grid::cell_idx_t c10(i1, j0, kk, lb);
            spade::grid::cell_idx_t c01(i0, j1, kk, lb);
            spade::grid::cell_idx_t c11(i1, j1, kk, lb);
    
            const state_type q00 = arr_img.get_elem(c00);
            const state_type q10 = arr_img.get_elem(c10);
            const state_type q01 = arr_img.get_elem(c01);
            const state_type q11 = arr_img.get_elem(c11);
    
            // Bilinear interpolation
            const state_type qx0 = (prec_type(1.0) - tx)*q00 + tx*q10;
            const state_type qx1 = (prec_type(1.0) - tx)*q01 + tx*q11;
            const state_type q   = (prec_type(1.0) - ty)*qx0 + ty*qx1;
            
            float value = q.u();
            auto c = cmap.get_color(value, vmin, vmax);
            
            r_out = c[0];
            g_out = c[1];
            b_out = c[2];
            
            gimg.set_elem(row, col, g_out);
            bimg.set_elem(row, col, b_out);
            return r_out;
        });
        
        spade::ext::gfx::output_image(fname, r, g, b);
    }
}