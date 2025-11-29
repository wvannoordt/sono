#pragma once

#include "scidf.h"
#include "spade.h"
#include "solver.h"

namespace sl
{
    struct sample
    {
        using coords_type = double;
        using point_type  = spade::coords::point_t<coords_type>;
        point_type x;
        std::size_t lb; // which block?
    };
    
    struct sample_stream
    {
        using value_type = typename solver::state_type;
        spade::device::shared_vector<value_type> data_buf;

        /// Construct a CSV stream.
        ///  - filename: path to CSV file
        ///  - headers:  column names
        ///  - append:   true -> append to existing file, false -> overwrite
        sample_stream(const std::string& filename,
                        const std::vector<std::string>& headers,
                        bool append = true)
        : ncols_(headers.size())
        {
            if (ncols_ == 0)
            {
                throw std::runtime_error("sample_stream: header list must not be empty.");
            }

            std::ios_base::openmode mode = std::ios::out;
            if (append) mode |= std::ios::app;
            else        mode |= std::ios::trunc;

            // Decide whether to write header (if appending and file already
            // has content, skip header).
            bool write_header = true;
            if (append)
            {
                std::ifstream fin(filename);
                if (fin.good())
                {
                    fin.seekg(0, std::ios::end);
                    if (fin.tellg() > 0) write_header = false;
                }
            }

            file_.open(filename, mode);
            if (!file_)
            {
                throw std::runtime_error("sample_stream: failed to open file '" + filename + "'.");
            }

            if (write_header)
            {
                write_header_line(headers);
            }
        }

        // No copying (streams don't copy well)
        sample_stream(const sample_stream&) = delete;
        sample_stream& operator=(const sample_stream&) = delete;

        // Moving is fine if you ever need it
        sample_stream(sample_stream&&) = default;
        sample_stream& operator=(sample_stream&&) = default;

        ~sample_stream()
        {
            if (file_.is_open()) file_.close();
        }

        /// Append one row of data.
        /// Size of `row` must match the number of headers.
        void write_row(const std::vector<value_type>& row)
        {
            if (!file_)
            {
                throw std::runtime_error("sample_stream: file stream is not valid.");
            }
            if (row.size() != ncols_)
            {
                throw std::runtime_error("sample_stream: row column count does not match header count.");
            }

            for (std::size_t i = 0; i < row.size(); ++i)
            {
                if (i) file_ << ',';
                file_ << row[i].u();
            }
            file_ << '\n';
            file_.flush();
        }

    private:
        std::ofstream file_;
        std::size_t   ncols_;

        void write_header_line(const std::vector<std::string>& headers)
        {
            for (std::size_t i = 0; i < headers.size(); ++i)
            {
                if (i) file_ << ',';
                file_ << headers[i];
            }
            file_ << '\n';
            file_.flush();
        }
    };
    
    inline void write_samples(sample_stream& output, const spade::device::shared_vector<sample>& samps, const typename solver::array_type& arr)
    {
        if (output.data_buf.size() != samps.size()) { output.data_buf.resize(samps.size()); output.data_buf.transfer(); }
        const auto grid_img = arr.get_grid().image(spade::partition::local, arr.device());
        const auto arr_img  = arr.image();
        using state_type = typename sample_stream::value_type;
        const auto samps_img = spade::utils::make_vec_image(samps.data(arr.device()));
        auto data_img = spade::utils::make_vec_image(output.data_buf.data(arr.device()));
        using prec_type = typename state_type::value_type;
        auto lam = [=] _sp_hybrid (const std::size_t& j) mutable
        {
            const auto lb   = samps_img[j].lb;
            const auto& bnd = grid_img.get_bounding_box(lb);
            const auto& dx  = grid_img.get_dx(lb);
    
            // Sample position
            const auto pos = samps_img[j].x;
            const double x = pos[0];
            const double y = pos[1];
    
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
    
            // Write out interpolated value for this sample
            data_img[j] = q;
        };
            
        spade::dispatch::execute(
            spade::dispatch::ranges::make_range(0UL, samps.size()),
            lam,
            arr.device()
        );
        output.data_buf.itransfer();
        output.write_row(output.data_buf.data(spade::device::cpu));
    }
    
    inline std::pair<std::vector<std::string>, spade::device::shared_vector<sample>> read_samples(scidf::node_t& sec, const solver& slvr)
    {
        spade::device::shared_vector<sample> out;
        std::vector<std::string> names;
        for (auto& [nm, cd] : sec.children)
        {
            spade::ctrs::array<double, 2> p = cd;
            sample news;
            news.x[0] = p[0];
            news.x[1] = p[1];
            news.x[2] = 0;
            news.lb   = 0;
            out.push_back(news);
            names.push_back(nm);
        }
        
        auto bv = slvr.get_grid().compute_bvh(spade::partition::local);
        for (std::size_t j = 0; j < out.size(); ++j)
        {
            const auto dst = [&](const auto& xx, const std::size_t& ilb)
            {
                const auto bbx = slvr.get_grid().get_bounding_box(spade::utils::tag[spade::partition::local](ilb));
                return bv.box_min_dist(bbx, xx);
            };
            typename std::decay_t<decltype(bv)>::pnt_t xx(out[j].x[0], out[j].x[1], 0);
            out[j].lb = bv.find_closest_element(xx, dst);
        }
        
        out.transfer();
        return {names, out};
    }
}