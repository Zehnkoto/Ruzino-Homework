#pragma once
#include <exprtk/exprtk.hpp>
#include <stdexcept>

#include "pxr/base/gf/vec3f.h"
namespace USTC_CG {

namespace fem_bem {

    // Multi-variable Simpson integration on simplex using barycentric
    // coordinates
    template<typename T>
    inline T integrate_simplex(
        const exprtk::expression<T>& e,
        const std::vector<std::string>& barycentric_names,
        const std::size_t number_of_intervals = 100)
    {
        if (barycentric_names.empty())
            return T(0);

        const std::size_t dim = barycentric_names.size();
        const exprtk::symbol_table<T>& sym_table = e.get_symbol_table();

        if (!sym_table.valid())
            return std::numeric_limits<T>::quiet_NaN();

        // Get variable references
        std::vector<exprtk::details::variable_node<T>*> vars(dim);
        std::vector<T> original_values(dim);

        for (std::size_t i = 0; i < dim; ++i) {
            vars[i] = sym_table.get_variable(barycentric_names[i]);
            if (!vars[i])
                return std::numeric_limits<T>::quiet_NaN();
            original_values[i] = vars[i]->ref();
        }

        T total_integral = T(0);

        // For 1D case (line segment)
        if (dim == 1) {
            const T h = T(1) / number_of_intervals;

            for (std::size_t i = 0; i < number_of_intervals; ++i) {
                const T u1 = i * h;
                const T u2 = (i + 1) * h;
                const T u_mid = (u1 + u2) / T(2);

                // Simpson's rule over [u1, u2]
                vars[0]->ref() = u1;
                const T y1 = e.value();

                vars[0]->ref() = u_mid;
                const T y_mid = e.value();

                vars[0]->ref() = u2;
                const T y2 = e.value();

                total_integral += h * (y1 + T(4) * y_mid + y2) / T(6);
            }
        }
        // For 2D case (triangle) - uses 2 barycentric coordinates
        else if (dim == 2) {
            const T h = T(1) / number_of_intervals;

            for (std::size_t i = 0; i <= number_of_intervals; ++i) {
                for (std::size_t j = 0; j <= number_of_intervals - i; ++j) {
                    const T u1 = i * h;
                    const T u2 = j * h;

                    if (u1 + u2 <= T(1)) {
                        vars[0]->ref() = u1;
                        vars[1]->ref() = u2;

                        T weight = T(1);
                        // Boundary correction for trapezoidal rule
                        if (i == 0 || j == 0 || i + j == number_of_intervals)
                            weight = T(0.5);
                        if ((i == 0 && j == 0) ||
                            (i == 0 && i + j == number_of_intervals) ||
                            (j == 0 && i + j == number_of_intervals))
                            weight = T(0.25);

                        // Triangle area element: h*h/2 * 2 = h*h
                        total_integral += weight * e.value() * h * h * T(2);
                    }
                }
            }
        }
        // For 3D case (tetrahedron) - uses 3 barycentric coordinates  
        else if (dim == 3) {
            const T h = T(1) / number_of_intervals;

            for (std::size_t i = 0; i <= number_of_intervals; ++i) {
                for (std::size_t j = 0; j <= number_of_intervals - i; ++j) {
                    for (std::size_t k = 0; k <= number_of_intervals - i - j; ++k) {
                        const T u1 = i * h;
                        const T u2 = j * h;
                        const T u3 = k * h;

                        if (u1 + u2 + u3 <= T(1)) {
                            vars[0]->ref() = u1;
                            vars[1]->ref() = u2;
                            vars[2]->ref() = u3;

                            T weight = T(1);
                            // Boundary correction
                            int boundary_count = 0;
                            if (i == 0) boundary_count++;
                            if (j == 0) boundary_count++;
                            if (k == 0) boundary_count++;
                            if (i + j + k == number_of_intervals) boundary_count++;

                            if (boundary_count > 0)
                                weight = T(1) / T(1 << boundary_count);

                            // Tetrahedron volume element: multiply by 6 to normalize to unit volume
                            total_integral += weight * e.value() * h * h * h * T(6);
                        }
                    }
                }
            }
        }

        // Restore original values
        for (std::size_t i = 0; i < dim; ++i) {
            vars[i]->ref() = original_values[i];
        }

        return total_integral;
    }

    // Integration of product of two expressions over simplex using barycentric
    // coordinates
    template<typename T>
    inline T integrate_product_simplex(
        const exprtk::expression<T>& expr1,
        const exprtk::expression<T>& expr2,
        const std::vector<std::string>& barycentric_names,
        const std::size_t number_of_intervals = 100)
    {
        if (barycentric_names.empty())
            return T(0);

        const std::size_t dim = barycentric_names.size();
        const exprtk::symbol_table<T>& sym_table1 = expr1.get_symbol_table();
        const exprtk::symbol_table<T>& sym_table2 = expr2.get_symbol_table();

        if (!sym_table1.valid() || !sym_table2.valid())
            return std::numeric_limits<T>::quiet_NaN();

        // Get variable references for expr1 (shape function)
        std::vector<exprtk::details::variable_node<T>*> vars1(dim);
        std::vector<T> original_values1(dim);

        for (std::size_t i = 0; i < dim; ++i) {
            vars1[i] = sym_table1.get_variable(barycentric_names[i]);
            if (!vars1[i])
                return std::numeric_limits<T>::quiet_NaN();
            original_values1[i] = vars1[i]->ref();
        }

        // Get variable references for expr2 (integrand)
        std::vector<exprtk::details::variable_node<T>*> vars2(dim);
        std::vector<T> original_values2(dim);

        for (std::size_t i = 0; i < dim; ++i) {
            vars2[i] = sym_table2.get_variable(barycentric_names[i]);
            if (!vars2[i])
                return std::numeric_limits<T>::quiet_NaN();
            original_values2[i] = vars2[i]->ref();
        }

        T total_integral = T(0);

        // For 1D case (line segment)
        if (dim == 1) {
            const T h = T(1) / number_of_intervals;

            for (std::size_t i = 0; i < number_of_intervals; ++i) {
                const T u1 = i * h;
                const T u2 = (i + 1) * h;
                const T u_mid = (u1 + u2) / T(2);

                // Simpson's rule over [u1, u2]
                vars1[0]->ref() = u1;
                vars2[0]->ref() = u1;
                const T y1 = expr1.value() * expr2.value();

                vars1[0]->ref() = u_mid;
                vars2[0]->ref() = u_mid;
                const T y_mid = expr1.value() * expr2.value();

                vars1[0]->ref() = u2;
                vars2[0]->ref() = u2;
                const T y2 = expr1.value() * expr2.value();

                total_integral += h * (y1 + T(4) * y_mid + y2) / T(6);
            }
        }
        // For 2D case (triangle)
        else if (dim == 2) {
            const T h = T(1) / number_of_intervals;

            for (std::size_t i = 0; i <= number_of_intervals; ++i) {
                for (std::size_t j = 0; j <= number_of_intervals - i; ++j) {
                    const T u1 = i * h;
                    const T u2 = j * h;

                    if (u1 + u2 <= T(1)) {
                        // Set barycentric coordinates for both expressions
                        vars1[0]->ref() = u1;
                        vars2[0]->ref() = u1;
                        vars1[1]->ref() = u2;
                        vars2[1]->ref() = u2;

                        T weight = T(1);
                        // Boundary correction for trapezoidal rule
                        if (i == 0 || j == 0 || i + j == number_of_intervals)
                            weight = T(0.5);
                        if ((i == 0 && j == 0) ||
                            (i == 0 && i + j == number_of_intervals) ||
                            (j == 0 && i + j == number_of_intervals))
                            weight = T(0.25);

                        // Product of the two expressions
                        T product_value = expr1.value() * expr2.value();
                        total_integral += weight * product_value * h * h * T(2);
                    }
                }
            }
        }
        // For 3D case (tetrahedron)
        else if (dim == 3) {
            const T h = T(1) / number_of_intervals;

            for (std::size_t i = 0; i <= number_of_intervals; ++i) {
                for (std::size_t j = 0; j <= number_of_intervals - i; ++j) {
                    for (std::size_t k = 0; k <= number_of_intervals - i - j;
                         ++k) {
                        const T u1 = i * h;
                        const T u2 = j * h;
                        const T u3 = k * h;

                        if (u1 + u2 + u3 <= T(1)) {
                            vars1[0]->ref() = u1;
                            vars2[0]->ref() = u1;
                            vars1[1]->ref() = u2;
                            vars2[1]->ref() = u2;
                            vars1[2]->ref() = u3;
                            vars2[2]->ref() = u3;

                            T weight = T(1);
                            // Boundary correction
                            int boundary_count = 0;
                            if (i == 0)
                                boundary_count++;
                            if (j == 0)
                                boundary_count++;
                            if (k == 0)
                                boundary_count++;
                            if (i + j + k == number_of_intervals)
                                boundary_count++;

                            if (boundary_count > 0)
                                weight = T(1) / T(1 << boundary_count);

                            T product_value = expr1.value() * expr2.value();
                            total_integral +=
                                weight * product_value * h * h * h * T(6);
                        }
                    }
                }
            }
        }

        // Restore original values for both expressions
        for (std::size_t i = 0; i < dim; ++i) {
            vars1[i]->ref() = original_values1[i];
            vars2[i]->ref() = original_values2[i];
        }

        return total_integral;
    }

    // Integration of product with mapping function support
    // Integrates expr1(u) * expr2(pull_back(u)) over simplex where u are
    // barycentric coordinates
    template<typename T, typename MappingFunc>
    inline T integrate_product_simplex_with_mapping(
        const exprtk::expression<T>& expr1,
        const exprtk::expression<T>& expr2,
        MappingFunc mapping,
        const std::vector<std::string>& barycentric_names,
        const std::size_t number_of_intervals = 100)
    {
        if (barycentric_names.empty())
            return T(0);

        const std::size_t dim = barycentric_names.size();
        const exprtk::symbol_table<T>& sym_table1 = expr1.get_symbol_table();
        const exprtk::symbol_table<T>& sym_table2 = expr2.get_symbol_table();

        if (!sym_table1.valid() || !sym_table2.valid())
            return std::numeric_limits<T>::quiet_NaN();

        // Get variable references for barycentric coordinates in both
        // expressions
        std::vector<exprtk::details::variable_node<T>*> barycentric_vars1(dim);
        std::vector<exprtk::details::variable_node<T>*> barycentric_vars2(dim);
        std::vector<T> original_values1(dim), original_values2(dim);

        for (std::size_t i = 0; i < dim; ++i) {
            barycentric_vars1[i] =
                sym_table1.get_variable(barycentric_names[i]);
            barycentric_vars2[i] =
                sym_table2.get_variable(barycentric_names[i]);
            if (!barycentric_vars1[i] || !barycentric_vars2[i])
                return std::numeric_limits<T>::quiet_NaN();
            original_values1[i] = barycentric_vars1[i]->ref();
            original_values2[i] = barycentric_vars2[i]->ref();
        }

        // Check for physical coordinate variables in expr2 symbol table
        exprtk::details::variable_node<T>* x_var = sym_table2.get_variable("x");
        exprtk::details::variable_node<T>* y_var = sym_table2.get_variable("y");
        exprtk::details::variable_node<T>* z_var = sym_table2.get_variable("z");

        T x_orig = x_var ? x_var->ref() : T(0);
        T y_orig = y_var ? y_var->ref() : T(0);
        T z_orig = z_var ? z_var->ref() : T(0);

        T total_integral = T(0);

        // For 1D case (line segment)
        if (dim == 1) {
            const T h = T(1) / number_of_intervals;

            for (std::size_t i = 0; i < number_of_intervals; ++i) {
                const T u1 = i * h;
                const T u2 = (i + 1) * h;
                const T u_mid = (u1 + u2) / T(2);

                // Simpson's rule over [u1, u2]
                barycentric_vars1[0]->ref() = u1;
                barycentric_vars2[0]->ref() = u1;
                auto mapped_coords = mapping(u1);
                if (x_var && mapped_coords.size() > 0)
                    x_var->ref() = mapped_coords[0];
                if (y_var && mapped_coords.size() > 1)
                    y_var->ref() = mapped_coords[1];
                if (z_var && mapped_coords.size() > 2)
                    z_var->ref() = mapped_coords[2];
                const T y1 = expr1.value() * expr2.value();

                barycentric_vars1[0]->ref() = u_mid;
                barycentric_vars2[0]->ref() = u_mid;
                mapped_coords = mapping(u_mid);
                if (x_var && mapped_coords.size() > 0)
                    x_var->ref() = mapped_coords[0];
                if (y_var && mapped_coords.size() > 1)
                    y_var->ref() = mapped_coords[1];
                if (z_var && mapped_coords.size() > 2)
                    z_var->ref() = mapped_coords[2];
                const T y_mid = expr1.value() * expr2.value();

                barycentric_vars1[0]->ref() = u2;
                barycentric_vars2[0]->ref() = u2;
                mapped_coords = mapping(u2);
                if (x_var && mapped_coords.size() > 0)
                    x_var->ref() = mapped_coords[0];
                if (y_var && mapped_coords.size() > 1)
                    y_var->ref() = mapped_coords[1];
                if (z_var && mapped_coords.size() > 2)
                    z_var->ref() = mapped_coords[2];
                const T y2 = expr1.value() * expr2.value();

                total_integral += h * (y1 + T(4) * y_mid + y2) / T(6);
            }
        }
        // For 2D case (triangle)
        else if (dim == 2) {
            const T h = T(1) / number_of_intervals;

            for (std::size_t i = 0; i <= number_of_intervals; ++i) {
                for (std::size_t j = 0; j <= number_of_intervals - i; ++j) {
                    const T u1 = i * h;
                    const T u2 = j * h;

                    if (u1 + u2 <= T(1)) {
                        // Set barycentric coordinates for both expressions
                        barycentric_vars1[0]->ref() = u1;
                        barycentric_vars2[0]->ref() = u1;
                        barycentric_vars1[1]->ref() = u2;
                        barycentric_vars2[1]->ref() = u2;

                        // Apply mapping function to get physical coordinates for expr2
                        auto mapped_coords = mapping(u1, u2);

                        // Update physical coordinate variables if they exist in expr2
                        if (x_var && mapped_coords.size() > 0)
                            x_var->ref() = mapped_coords[0];
                        if (y_var && mapped_coords.size() > 1)
                            y_var->ref() = mapped_coords[1];
                        if (z_var && mapped_coords.size() > 2)
                            z_var->ref() = mapped_coords[2];

                        T weight = T(1);
                        // Boundary correction for trapezoidal rule
                        if (i == 0 || j == 0 || i + j == number_of_intervals)
                            weight = T(0.5);
                        if ((i == 0 && j == 0) ||
                            (i == 0 && i + j == number_of_intervals) ||
                            (j == 0 && i + j == number_of_intervals))
                            weight = T(0.25);

                        // Product: shape_func(u) * expr(pull_back(u))
                        T product_value = expr1.value() * expr2.value();
                        total_integral += weight * product_value * h * h * T(2);
                    }
                }
            }
        }
        // For 3D case (tetrahedron)
        else if (dim == 3) {
            const T h = T(1) / number_of_intervals;

            for (std::size_t i = 0; i <= number_of_intervals; ++i) {
                for (std::size_t j = 0; j <= number_of_intervals - i; ++j) {
                    for (std::size_t k = 0; k <= number_of_intervals - i - j;
                         ++k) {
                        const T u1 = i * h;
                        const T u2 = j * h;
                        const T u3 = k * h;

                        if (u1 + u2 + u3 <= T(1)) {
                            barycentric_vars1[0]->ref() = u1;
                            barycentric_vars2[0]->ref() = u1;
                            barycentric_vars1[1]->ref() = u2;
                            barycentric_vars2[1]->ref() = u2;
                            barycentric_vars1[2]->ref() = u3;
                            barycentric_vars2[2]->ref() = u3;

                            auto mapped_coords = mapping(u1, u2, u3);
                            if (x_var && mapped_coords.size() > 0)
                                x_var->ref() = mapped_coords[0];
                            if (y_var && mapped_coords.size() > 1)
                                y_var->ref() = mapped_coords[1];
                            if (z_var && mapped_coords.size() > 2)
                                z_var->ref() = mapped_coords[2];

                            T weight = T(1);
                            // Boundary correction
                            int boundary_count = 0;
                            if (i == 0)
                                boundary_count++;
                            if (j == 0)
                                boundary_count++;
                            if (k == 0)
                                boundary_count++;
                            if (i + j + k == number_of_intervals)
                                boundary_count++;

                            if (boundary_count > 0)
                                weight = T(1) / T(1 << boundary_count);

                            T product_value = expr1.value() * expr2.value();
                            total_integral +=
                                weight * product_value * h * h * h * T(6);
                        }
                    }
                }
            }
        }

        // Restore original values
        for (std::size_t i = 0; i < dim; ++i) {
            barycentric_vars1[i]->ref() = original_values1[i];
            barycentric_vars2[i]->ref() = original_values2[i];
        }
        if (x_var)
            x_var->ref() = x_orig;
        if (y_var)
            y_var->ref() = y_orig;
        if (z_var)
            z_var->ref() = z_orig;

        return total_integral;
    }

}  // namespace fem_bem

}  // namespace USTC_CG
