#pragma once
#include <exprtk/exprtk.hpp>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "GCore/GOP.h"
#include "integrate.hpp"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec3d.h"
USTC_CG_NAMESPACE_OPEN_SCOPE

namespace fem_bem {

enum class ElementBasisType { FiniteElement, BoundaryElement };

// Base template for element basis with dimension-aware expression storage
template<unsigned ProblemDimension, ElementBasisType Type, typename T = double>
class ElementBasis {
   public:
    using value_type = T;
    using expression_type = exprtk::expression<T>;
    using symbol_table_type = exprtk::symbol_table<T>;
    using parser_type = exprtk::parser<T>;

    static constexpr unsigned problem_dimension = ProblemDimension;
    static constexpr unsigned element_dimension =
        (Type == ElementBasisType::FiniteElement) ? ProblemDimension
                                                  : ProblemDimension - 1;
    static constexpr ElementBasisType type = Type;

    static_assert(
        ProblemDimension >= 1 && ProblemDimension <= 3,
        "Problem dimension must be 1, 2, or 3");
    static_assert(
        Type == ElementBasisType::BoundaryElement ? ProblemDimension >= 2
                                                  : true,
        "Boundary elements require problem dimension >= 2");

    ElementBasis()
    {
        // Initialize barycentric coordinate variables
        setup_barycentric_variables();
    }
    virtual ~ElementBasis() = default;

    // Vertex expressions (always available) - 0D knots
    void add_vertex_expression(const std::string& expr)
    {
        vertex_expressions_.push_back(expr);
        vertex_parsed_.clear();  // Clear parsed cache
    }
    void set_vertex_expressions(const std::vector<std::string>& exprs)
    {
        vertex_expressions_ = exprs;
        vertex_parsed_.clear();  // Clear parsed cache
    }
    const std::vector<std::string>& get_vertex_expressions() const
    {
        return vertex_expressions_;
    }
    // Get parsed expressions, parse on demand
    const std::vector<expression_type>& get_parsed_vertex_expressions() const
    {
        if (vertex_parsed_.empty() && !vertex_expressions_.empty()) {
            parse_vertex_expressions();
        }
        return vertex_parsed_;
    }
    void clear_vertex_expressions()
    {
        vertex_expressions_.clear();
        vertex_parsed_.clear();
    }

    // Edge expressions (available when element_dimension >= 2) - 1D knots
    void add_edge_expression(const std::string& expr)
    {
        if constexpr (element_dimension >= 2) {
            edge_expressions_.push_back(expr);
            edge_parsed_.clear();
        }
    }

    void set_edge_expressions(const std::vector<std::string>& exprs)
    {
        if constexpr (element_dimension >= 2) {
            edge_expressions_ = exprs;
            edge_parsed_.clear();
        }
    }

    const std::vector<std::string>& get_edge_expressions() const
    {
        if constexpr (element_dimension >= 2) {
            return edge_expressions_;
        }
        else {
            static const std::vector<std::string> empty;
            return empty;
        }
    }

    const std::vector<expression_type>& get_parsed_edge_expressions() const
    {
        if constexpr (element_dimension >= 2) {
            if (edge_parsed_.empty() && !edge_expressions_.empty()) {
                parse_edge_expressions();
            }
            return edge_parsed_;
        }
        else {
            static const std::vector<expression_type> empty;
            return empty;
        }
    }

    void clear_edge_expressions()
    {
        if constexpr (element_dimension >= 2) {
            edge_expressions_.clear();
            edge_parsed_.clear();
        }
    }

    // Face expressions (available when element_dimension >= 3) - 2D knots
    void add_face_expression(const std::string& expr)
    {
        if constexpr (element_dimension >= 3) {
            face_expressions_.push_back(expr);
            face_parsed_.clear();
        }
    }

    void set_face_expressions(const std::vector<std::string>& exprs)
    {
        if constexpr (element_dimension >= 3) {
            face_expressions_ = exprs;
            face_parsed_.clear();
        }
    }

    const std::vector<std::string>& get_face_expressions() const
    {
        if constexpr (element_dimension >= 3) {
            return face_expressions_;
        }
        else {
            static const std::vector<std::string> empty;
            return empty;
        }
    }

    const std::vector<expression_type>& get_parsed_face_expressions() const
    {
        if constexpr (element_dimension >= 3) {
            if (face_parsed_.empty() && !face_expressions_.empty()) {
                parse_face_expressions();
            }
            return face_parsed_;
        }
        else {
            static const std::vector<expression_type> empty;
            return empty;
        }
    }

    void clear_face_expressions()
    {
        if constexpr (element_dimension >= 3) {
            face_expressions_.clear();
            face_parsed_.clear();
        }
    }

    // Volume expressions (only for 3D elements) - 3D knots
    void add_volume_expression(const std::string& expr)
    {
        if constexpr (element_dimension == 3) {
            volume_expressions_.push_back(expr);
            volume_parsed_.clear();
        }
    }

    void set_volume_expressions(const std::vector<std::string>& exprs)
    {
        if constexpr (element_dimension == 3) {
            volume_expressions_ = exprs;
            volume_parsed_.clear();
        }
    }

    const std::vector<std::string>& get_volume_expressions() const
    {
        if constexpr (element_dimension == 3) {
            return volume_expressions_;
        }
        else {
            static const std::vector<std::string> empty;
            return empty;
        }
    }

    const std::vector<expression_type>& get_parsed_volume_expressions() const
    {
        if constexpr (element_dimension == 3) {
            if (volume_parsed_.empty() && !volume_expressions_.empty()) {
                parse_volume_expressions();
            }
            return volume_parsed_;
        }
        else {
            static const std::vector<expression_type> empty;
            return empty;
        }
    }

    void clear_volume_expressions()
    {
        if constexpr (element_dimension == 3) {
            volume_expressions_.clear();
            volume_parsed_.clear();
        }
    }

    // Check if specific expression types are available at compile time
    static constexpr bool has_edge_expressions()
    {
        return element_dimension >= 2;
    }
    static constexpr bool has_face_expressions()
    {
        return element_dimension >= 3;
    }
    static constexpr bool has_volume_expressions()
    {
        return element_dimension == 3;
    }

    // Integration interface: integrate shape functions against expressions

    // Integrate vertex shape functions against expression
    template<typename MappingFunc = std::nullptr_t>
    std::vector<T> integrate_vertex_against(
        const std::string& expr_str,
        MappingFunc mapping = nullptr,
        std::size_t intervals = 100) const
    {
        std::vector<T> results;
        const auto& vertex_exprs = get_parsed_vertex_expressions();
        results.reserve(vertex_exprs.size());

        for (const auto& shape_func : vertex_exprs) {
            results.push_back(integrate_shape_function_against_expression(
                shape_func, expr_str, mapping, intervals));
        }
        return results;
    }

    // Integrate edge shape functions against expression (only for
    // element_dimension >= 2)
    template<typename MappingFunc = std::nullptr_t>
    std::vector<T> integrate_edge_against(
        const std::string& expr_str,
        MappingFunc mapping = nullptr,
        std::size_t intervals = 100) const
    {
        if constexpr (element_dimension >= 2) {
            std::vector<T> results;
            const auto& edge_exprs = get_parsed_edge_expressions();
            results.reserve(edge_exprs.size());

            for (const auto& shape_func : edge_exprs) {
                results.push_back(integrate_shape_function_against_expression(
                    shape_func, expr_str, mapping, intervals));
            }
            return results;
        }
        else {
            return std::vector<T>();
        }
    }

    // Integrate face shape functions against expression (only for
    // element_dimension >= 3)
    template<typename MappingFunc = std::nullptr_t>
    std::vector<T> integrate_face_against(
        const std::string& expr_str,
        MappingFunc mapping = nullptr,
        std::size_t intervals = 100) const
    {
        if constexpr (element_dimension >= 3) {
            std::vector<T> results;
            const auto& face_exprs = get_parsed_face_expressions();
            results.reserve(face_exprs.size());

            for (const auto& shape_func : face_exprs) {
                results.push_back(integrate_shape_function_against_expression(
                    shape_func, expr_str, mapping, intervals));
            }
            return results;
        }
        else {
            return std::vector<T>();
        }
    }

    // Integrate volume shape functions against expression (only for
    // element_dimension == 3)
    template<typename MappingFunc = std::nullptr_t>
    std::vector<T> integrate_volume_against(
        const std::string& expr_str,
        MappingFunc mapping = nullptr,
        std::size_t intervals = 100) const
    {
        if constexpr (element_dimension == 3) {
            std::vector<T> results;
            const auto& volume_exprs = get_parsed_volume_expressions();
            results.reserve(volume_exprs.size());

            for (const auto& shape_func : volume_exprs) {
                results.push_back(integrate_shape_function_against_expression(
                    shape_func, expr_str, mapping, intervals));
            }
            return results;
        }
        else {
            return std::vector<T>();
        }
    }

   private:
    // Setup barycentric coordinate variables based on element dimension
    void setup_barycentric_variables()
    {
        if constexpr (element_dimension == 1) {
            barycentric_names_ = { "u1" };
            barycentric_vars_.resize(1);
        }
        else if constexpr (element_dimension == 2) {
            barycentric_names_ = { "u1", "u2" };
            barycentric_vars_.resize(2);
        }
        else if constexpr (element_dimension == 3) {
            barycentric_names_ = { "u1", "u2", "u3" };
            barycentric_vars_.resize(3);
        }

        // Add variables to symbol table
        for (std::size_t i = 0; i < barycentric_names_.size(); ++i) {
            barycentric_vars_[i] = T(0);
            symbol_table_.add_variable(
                barycentric_names_[i], barycentric_vars_[i]);
        }
    }

    // Parse expressions with current symbol table
    void parse_vertex_expressions() const
    {
        vertex_parsed_.clear();
        vertex_parsed_.reserve(vertex_expressions_.size());

        parser_type parser;
        for (const auto& expr_str : vertex_expressions_) {
            expression_type expr;
            expr.register_symbol_table(symbol_table_);

            if (parser.compile(expr_str, expr)) {
                vertex_parsed_.push_back(std::move(expr));
            }
            else {
                throw std::runtime_error(
                    "Failed to parse vertex expression: " + expr_str);
            }
        }
    }

    template<typename U = void>
    void parse_edge_expressions() const
    {
        if constexpr (element_dimension >= 2) {
            edge_parsed_.clear();
            edge_parsed_.reserve(edge_expressions_.size());

            parser_type parser;
            for (const auto& expr_str : edge_expressions_) {
                expression_type expr;
                expr.register_symbol_table(symbol_table_);

                if (parser.compile(expr_str, expr)) {
                    edge_parsed_.push_back(std::move(expr));
                }
                else {
                    throw std::runtime_error(
                        "Failed to parse edge expression: " + expr_str);
                }
            }
        }
    }

    template<typename U = void>
    void parse_face_expressions() const
    {
        if constexpr (element_dimension >= 3) {
            face_parsed_.clear();
            face_parsed_.reserve(face_expressions_.size());

            parser_type parser;
            for (const auto& expr_str : face_expressions_) {
                expression_type expr;
                expr.register_symbol_table(symbol_table_);

                if (parser.compile(expr_str, expr)) {
                    face_parsed_.push_back(std::move(expr));
                }
                else {
                    throw std::runtime_error(
                        "Failed to parse face expression: " + expr_str);
                }
            }
        }
    }

    template<typename U = void>
    void parse_volume_expressions() const
    {
        if constexpr (element_dimension == 3) {
            volume_parsed_.clear();
            volume_parsed_.reserve(volume_expressions_.size());

            parser_type parser;
            for (const auto& expr_str : volume_expressions_) {
                expression_type expr;
                expr.register_symbol_table(symbol_table_);

                if (parser.compile(expr_str, expr)) {
                    volume_parsed_.push_back(std::move(expr));
                }
                else {
                    throw std::runtime_error(
                        "Failed to parse volume expression: " + expr_str);
                }
            }
        }
    }

    // Core integration implementation: integrates shape_function * expression
    // over simplex
    template<typename MappingFunc>
    T integrate_shape_function_against_expression(
        const expression_type& shape_func,
        const std::string& expr_str,
        MappingFunc mapping,
        std::size_t intervals) const
    {
        // Parse the expression to integrate against
        expression_type expr;
        symbol_table_type local_symbol_table =
            symbol_table_;  // Copy symbol table

        // Add x, y, z variables for world coordinates if mapping is provided
        T x_var = T(0), y_var = T(0), z_var = T(0);
        if constexpr (!std::is_same_v<MappingFunc, std::nullptr_t>) {
            local_symbol_table.add_variable("x", x_var);
            local_symbol_table.add_variable("y", y_var);
            local_symbol_table.add_variable("z", z_var);
        }

        expr.register_symbol_table(local_symbol_table);

        parser_type parser;
        if (!parser.compile(expr_str, expr)) {
            throw std::runtime_error("Failed to parse expression: " + expr_str);
        }

        // Create combined expression: shape_func * expr
        return integrate_product_with_mapping(
            shape_func, expr, mapping, intervals);
    }

    // Core integration of product of two expressions with mapping
    template<typename MappingFunc>
    T integrate_product_with_mapping(
        const expression_type& shape_func,
        const expression_type& expr,
        MappingFunc mapping,
        std::size_t intervals) const
    {
        if constexpr (std::is_same_v<MappingFunc, std::nullptr_t>) {
            // No mapping, integrate shape_func * expr directly
            return integrate_product_simplex(
                shape_func, expr, barycentric_names_, intervals);
        }
        else {
            // With mapping: integrate shape_func(u) * expr(pull_back(u)) over
            // barycentric coordinates
            return integrate_product_simplex_with_mapping(
                shape_func, expr, mapping, barycentric_names_, intervals);
        }
    }

   protected:
    // Symbol table and barycentric variables for expression parsing
    mutable symbol_table_type symbol_table_;
    std::vector<std::string> barycentric_names_;
    mutable std::vector<T> barycentric_vars_;

    // Vertex expressions are always available (0D knots)
    std::vector<std::string> vertex_expressions_;
    mutable std::vector<expression_type> vertex_parsed_;

    // Conditional member variables based on element dimension
    std::conditional_t<element_dimension >= 2, std::vector<std::string>, char>
        edge_expressions_;
    mutable std::conditional_t<
        element_dimension >= 2,
        std::vector<expression_type>,
        char>
        edge_parsed_;

    std::conditional_t<element_dimension >= 3, std::vector<std::string>, char>
        face_expressions_;
    mutable std::conditional_t<
        element_dimension >= 3,
        std::vector<expression_type>,
        char>
        face_parsed_;

    std::conditional_t<element_dimension == 3, std::vector<std::string>, char>
        volume_expressions_;
    mutable std::conditional_t<
        element_dimension == 3,
        std::vector<expression_type>,
        char>
        volume_parsed_;
};

// Convenient type aliases for common cases
template<unsigned ProblemDim>
using FiniteElementBasis =
    ElementBasis<ProblemDim, ElementBasisType::FiniteElement>;

template<unsigned ProblemDim>
using BoundaryElementBasis =
    ElementBasis<ProblemDim, ElementBasisType::BoundaryElement>;

// Specific instantiations with clear documentation
using FEM1D = FiniteElementBasis<1>;  // 1D finite elements (element_dim = 1)
using FEM2D = FiniteElementBasis<2>;  // 2D finite elements (element_dim = 2)
using FEM3D =
    FiniteElementBasis<3>;  // 3D finite elements (element_dim = 3, has volume)

using BEM2D = BoundaryElementBasis<2>;  // 1D boundary elements in 2D space
                                        // (element_dim = 1)
using BEM3D = BoundaryElementBasis<3>;  // 2D boundary elements in 3D space
                                        // (element_dim = 2)

// Base class for polymorphism when needed
class ElementBasisBase {
   public:
    virtual ~ElementBasisBase() = default;
    virtual ElementBasisType get_type() const = 0;
    virtual unsigned get_problem_dimension() const = 0;
    virtual unsigned get_element_dimension() const = 0;

    // Virtual interface for expressions
    virtual void add_vertex_expression(const std::string& expr) = 0;
    virtual const std::vector<std::string>& get_vertex_expressions() const = 0;

    // Virtual interface for integration without mapping
    virtual std::vector<double> integrate_vertex_against_str(
        const std::string& expr_str,
        std::size_t intervals = 100) const = 0;
    virtual std::vector<double> integrate_edge_against_str(
        const std::string& expr_str,
        std::size_t intervals = 100) const = 0;
    virtual std::vector<double> integrate_face_against_str(
        const std::string& expr_str,
        std::size_t intervals = 100) const = 0;
    virtual std::vector<double> integrate_volume_against_str(
        const std::string& expr_str,
        std::size_t intervals = 100) const = 0;

    // Virtual interface for integration with pullback mapping using world
    // coordinates
    virtual std::vector<double> integrate_vertex_against_with_mapping(
        const std::string& expr_str,
        const std::vector<pxr::GfVec2d>& world_vertices,
        std::size_t intervals = 100) const = 0;
    virtual std::vector<double> integrate_vertex_against_with_mapping(
        const std::string& expr_str,
        const std::vector<pxr::GfVec3d>& world_vertices,
        std::size_t intervals = 100) const = 0;

    virtual std::vector<double> integrate_edge_against_with_mapping(
        const std::string& expr_str,
        const std::vector<pxr::GfVec2d>& world_vertices,
        std::size_t intervals = 100) const = 0;
    virtual std::vector<double> integrate_edge_against_with_mapping(
        const std::string& expr_str,
        const std::vector<pxr::GfVec3d>& world_vertices,
        std::size_t intervals = 100) const = 0;

    virtual std::vector<double> integrate_face_against_with_mapping(
        const std::string& expr_str,
        const std::vector<pxr::GfVec3d>& world_vertices,
        std::size_t intervals = 100) const = 0;

    virtual std::vector<double> integrate_volume_against_with_mapping(
        const std::string& expr_str,
        const std::vector<pxr::GfVec3d>& world_vertices,
        std::size_t intervals = 100) const = 0;

    // Virtual interface for parsed expressions access
    virtual bool has_parsed_vertex_expressions() const = 0;
    virtual std::size_t vertex_expression_count() const = 0;

    // Dimension-specific virtual interfaces
    virtual bool supports_edge_expressions() const = 0;
    virtual bool supports_face_expressions() const = 0;
    virtual bool supports_volume_expressions() const = 0;

    // Edge expressions (only available if supported)
    virtual void add_edge_expression(const std::string& expr)
    {
    }
    virtual const std::vector<std::string>& get_edge_expressions() const
    {
        static const std::vector<std::string> empty;
        return empty;
    }
    virtual bool has_parsed_edge_expressions() const
    {
        return false;
    }
    virtual std::size_t edge_expression_count() const
    {
        return 0;
    }

    // Face expressions (only available if supported)
    virtual void add_face_expression(const std::string& expr)
    {
    }
    virtual const std::vector<std::string>& get_face_expressions() const
    {
        static const std::vector<std::string> empty;
        return empty;
    }
    virtual bool has_parsed_face_expressions() const
    {
        return false;
    }
    virtual std::size_t face_expression_count() const
    {
        return 0;
    }

    // Volume expressions (only available if supported)
    virtual void add_volume_expression(const std::string& expr)
    {
    }
    virtual const std::vector<std::string>& get_volume_expressions() const
    {
        static const std::vector<std::string> empty;
        return empty;
    }
    virtual bool has_parsed_volume_expressions() const
    {
        return false;
    }
    virtual std::size_t volume_expression_count() const
    {
        return 0;
    }
};

// Wrapper for type erasure when polymorphism is needed
template<unsigned ProblemDim, ElementBasisType Type>
class ElementBasisWrapper : public ElementBasisBase,
                            public ElementBasis<ProblemDim, Type> {
   public:
    ElementBasisType get_type() const override
    {
        return Type;
    }
    unsigned get_problem_dimension() const override
    {
        return ProblemDim;
    }
    unsigned get_element_dimension() const override
    {
        return ElementBasis<ProblemDim, Type>::element_dimension;
    }

    // Vertex expression interface
    void add_vertex_expression(const std::string& expr) override
    {
        ElementBasis<ProblemDim, Type>::add_vertex_expression(expr);
    }

    const std::vector<std::string>& get_vertex_expressions() const override
    {
        return ElementBasis<ProblemDim, Type>::get_vertex_expressions();
    }

    bool has_parsed_vertex_expressions() const override
    {
        auto& parsed = const_cast<ElementBasisWrapper*>(this)
                           ->get_parsed_vertex_expressions();
        return !parsed.empty();
    }

    std::size_t vertex_expression_count() const override
    {
        return ElementBasis<ProblemDim, Type>::get_vertex_expressions().size();
    }

    // Virtual integration interface implementations - without mapping
    std::vector<double> integrate_vertex_against_str(
        const std::string& expr_str,
        std::size_t intervals = 100) const override
    {
        auto results = ElementBasis<ProblemDim, Type>::integrate_vertex_against(
            expr_str, nullptr, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    std::vector<double> integrate_edge_against_str(
        const std::string& expr_str,
        std::size_t intervals = 100) const override
    {
        auto results = ElementBasis<ProblemDim, Type>::integrate_edge_against(
            expr_str, nullptr, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    std::vector<double> integrate_face_against_str(
        const std::string& expr_str,
        std::size_t intervals = 100) const override
    {
        auto results = ElementBasis<ProblemDim, Type>::integrate_face_against(
            expr_str, nullptr, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    std::vector<double> integrate_volume_against_str(
        const std::string& expr_str,
        std::size_t intervals = 100) const override
    {
        auto results = ElementBasis<ProblemDim, Type>::integrate_volume_against(
            expr_str, nullptr, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    // Virtual integration interface implementations - with mapping
    std::vector<double> integrate_vertex_against_with_mapping(
        const std::string& expr_str,
        const std::vector<pxr::GfVec2d>& world_vertices,
        std::size_t intervals = 100) const override
    {
        auto mapping = create_linear_mapping(world_vertices);
        auto results = ElementBasis<ProblemDim, Type>::integrate_vertex_against(
            expr_str, mapping, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    std::vector<double> integrate_vertex_against_with_mapping(
        const std::string& expr_str,
        const std::vector<pxr::GfVec3d>& world_vertices,
        std::size_t intervals = 100) const override
    {
        auto mapping = create_linear_mapping(world_vertices);
        auto results = ElementBasis<ProblemDim, Type>::integrate_vertex_against(
            expr_str, mapping, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    std::vector<double> integrate_edge_against_with_mapping(
        const std::string& expr_str,
        const std::vector<pxr::GfVec2d>& world_vertices,
        std::size_t intervals = 100) const override
    {
        auto mapping = create_linear_mapping(world_vertices);
        auto results = ElementBasis<ProblemDim, Type>::integrate_edge_against(
            expr_str, mapping, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    std::vector<double> integrate_edge_against_with_mapping(
        const std::string& expr_str,
        const std::vector<pxr::GfVec3d>& world_vertices,
        std::size_t intervals = 100) const override
    {
        auto mapping = create_linear_mapping(world_vertices);
        auto results = ElementBasis<ProblemDim, Type>::integrate_edge_against(
            expr_str, mapping, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    std::vector<double> integrate_face_against_with_mapping(
        const std::string& expr_str,
        const std::vector<pxr::GfVec3d>& world_vertices,
        std::size_t intervals = 100) const override
    {
        auto mapping = create_linear_mapping(world_vertices);
        auto results = ElementBasis<ProblemDim, Type>::integrate_face_against(
            expr_str, mapping, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    std::vector<double> integrate_volume_against_with_mapping(
        const std::string& expr_str,
        const std::vector<pxr::GfVec3d>& world_vertices,
        std::size_t intervals = 100) const override
    {
        auto mapping = create_linear_mapping(world_vertices);
        auto results = ElementBasis<ProblemDim, Type>::integrate_volume_against(
            expr_str, mapping, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

   private:
    // Helper functions to create linear mapping functions from world vertices
    auto create_linear_mapping(
        const std::vector<pxr::GfVec2d>& world_vertices) const
    {
        return [world_vertices](auto... u_values) -> std::vector<double> {
            std::vector<double> u_vec = { static_cast<double>(u_values)... };
            std::vector<double> result(2, 0.0);  // x, y coordinates

            // For 1D elements (BEM2D), we have 2 vertices and 1 barycentric coordinate u1
            // The second coordinate is u2 = 1 - u1
            if (u_vec.size() == 1 && world_vertices.size() >= 2) {
                double u1 = u_vec[0];
                double u2 = 1.0 - u1;
                result[0] = u1 * world_vertices[0][0] + u2 * world_vertices[1][0];  // x
                result[1] = u1 * world_vertices[0][1] + u2 * world_vertices[1][1];  // y
            }
            // For 2D elements (FEM2D), we have 3 vertices and 2 barycentric coordinates u1, u2
            // The third coordinate is u3 = 1 - u1 - u2  
            else if (u_vec.size() == 2 && world_vertices.size() >= 3) {
                double u1 = u_vec[0];
                double u2 = u_vec[1];
                double u3 = 1.0 - u1 - u2;
                result[0] = u1 * world_vertices[0][0] + u2 * world_vertices[1][0] + u3 * world_vertices[2][0];  // x
                result[1] = u1 * world_vertices[0][1] + u2 * world_vertices[1][1] + u3 * world_vertices[2][1];  // y
            }
            else {
                // Fallback for other cases
                for (size_t i = 0; i < world_vertices.size() && i < u_vec.size(); ++i) {
                    result[0] += u_vec[i] * world_vertices[i][0];  // x
                    result[1] += u_vec[i] * world_vertices[i][1];  // y
                }
            }
            return result;
        };
    }

    auto create_linear_mapping(
        const std::vector<pxr::GfVec3d>& world_vertices) const
    {
        return [world_vertices](auto... u_values) -> std::vector<double> {
            std::vector<double> u_vec = { static_cast<double>(u_values)... };
            std::vector<double> result(3, 0.0);  // x, y, z coordinates

            // For 1D elements (BEM2D), we have 2 vertices and 1 barycentric coordinate u1
            if (u_vec.size() == 1 && world_vertices.size() >= 2) {
                double u1 = u_vec[0];
                double u2 = 1.0 - u1;
                result[0] = u1 * world_vertices[0][0] + u2 * world_vertices[1][0];  // x
                result[1] = u1 * world_vertices[0][1] + u2 * world_vertices[1][1];  // y
                result[2] = u1 * world_vertices[0][2] + u2 * world_vertices[1][2];  // z
            }
            // For 2D elements (BEM3D), we have 3 vertices and 2 barycentric coordinates u1, u2
            else if (u_vec.size() == 2 && world_vertices.size() >= 3) {
                double u1 = u_vec[0];
                double u2 = u_vec[1];
                double u3 = 1.0 - u1 - u2;
                result[0] = u1 * world_vertices[0][0] + u2 * world_vertices[1][0] + u3 * world_vertices[2][0];  // x
                result[1] = u1 * world_vertices[0][1] + u2 * world_vertices[1][1] + u3 * world_vertices[2][1];  // y
                result[2] = u1 * world_vertices[0][2] + u2 * world_vertices[1][2] + u3 * world_vertices[2][2];  // z
            }
            // For 3D elements (FEM3D), we have 4 vertices and 3 barycentric coordinates u1, u2, u3
            else if (u_vec.size() == 3 && world_vertices.size() >= 4) {
                double u1 = u_vec[0];
                double u2 = u_vec[1]; 
                double u3 = u_vec[2];
                double u4 = 1.0 - u1 - u2 - u3;
                result[0] = u1 * world_vertices[0][0] + u2 * world_vertices[1][0] + u3 * world_vertices[2][0] + u4 * world_vertices[3][0];  // x
                result[1] = u1 * world_vertices[0][1] + u2 * world_vertices[1][1] + u3 * world_vertices[2][1] + u4 * world_vertices[3][1];  // y
                result[2] = u1 * world_vertices[0][2] + u2 * world_vertices[1][2] + u3 * world_vertices[2][2] + u4 * world_vertices[3][2];  // z
            }
            else {
                // Fallback for other cases
                for (size_t i = 0; i < world_vertices.size() && i < u_vec.size(); ++i) {
                    result[0] += u_vec[i] * world_vertices[i][0];  // x
                    result[1] += u_vec[i] * world_vertices[i][1];  // y
                    result[2] += u_vec[i] * world_vertices[i][2];  // z
                }
            }
            return result;
        };
    }

    // Template methods for custom mapping functions - These are the ONLY
    // integration interfaces
    template<typename MappingFunc = std::nullptr_t>
    std::vector<double> integrate_vertex_against(
        const std::string& expr_str,
        MappingFunc mapping = nullptr,
        std::size_t intervals = 100) const
    {
        auto results = ElementBasis<ProblemDim, Type>::integrate_vertex_against(
            expr_str, mapping, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    template<typename MappingFunc = std::nullptr_t>
    std::vector<double> integrate_edge_against(
        const std::string& expr_str,
        MappingFunc mapping = nullptr,
        std::size_t intervals = 100) const
    {
        auto results = ElementBasis<ProblemDim, Type>::integrate_edge_against(
            expr_str, mapping, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    template<typename MappingFunc = std::nullptr_t>
    std::vector<double> integrate_face_against(
        const std::string& expr_str,
        MappingFunc mapping = nullptr,
        std::size_t intervals = 100) const
    {
        auto results = ElementBasis<ProblemDim, Type>::integrate_face_against(
            expr_str, mapping, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    template<typename MappingFunc = std::nullptr_t>
    std::vector<double> integrate_volume_against(
        const std::string& expr_str,
        MappingFunc mapping = nullptr,
        std::size_t intervals = 100) const
    {
        auto results = ElementBasis<ProblemDim, Type>::integrate_volume_against(
            expr_str, mapping, intervals);
        return std::vector<double>(results.begin(), results.end());
    }

    // Dimension support checks
    bool supports_edge_expressions() const override
    {
        return ElementBasis<ProblemDim, Type>::has_edge_expressions();
    }

    bool supports_face_expressions() const override
    {
        return ElementBasis<ProblemDim, Type>::has_face_expressions();
    }

    bool supports_volume_expressions() const override
    {
        return ElementBasis<ProblemDim, Type>::has_volume_expressions();
    }

    // Edge expression interface (only if supported)
    void add_edge_expression(const std::string& expr) override
    {
        if constexpr (ElementBasis<ProblemDim, Type>::element_dimension >= 2) {
            ElementBasis<ProblemDim, Type>::add_edge_expression(expr);
        }
    }

    const std::vector<std::string>& get_edge_expressions() const override
    {
        if constexpr (ElementBasis<ProblemDim, Type>::element_dimension >= 2) {
            return ElementBasis<ProblemDim, Type>::get_edge_expressions();
        }
        else {
            static const std::vector<std::string> empty;
            return empty;
        }
    }

    bool has_parsed_edge_expressions() const override
    {
        if constexpr (ElementBasis<ProblemDim, Type>::element_dimension >= 2) {
            auto& parsed = const_cast<ElementBasisWrapper*>(this)
                               ->get_parsed_edge_expressions();
            return !parsed.empty();
        }
        return false;
    }

    std::size_t edge_expression_count() const override
    {
        if constexpr (ElementBasis<ProblemDim, Type>::element_dimension >= 2) {
            return ElementBasis<ProblemDim, Type>::get_edge_expressions()
                .size();
        }
        return 0;
    }

    // Face expression interface (only if supported)
    void add_face_expression(const std::string& expr) override
    {
        if constexpr (ElementBasis<ProblemDim, Type>::element_dimension >= 3) {
            ElementBasis<ProblemDim, Type>::add_face_expression(expr);
        }
    }

    const std::vector<std::string>& get_face_expressions() const override
    {
        if constexpr (ElementBasis<ProblemDim, Type>::element_dimension >= 3) {
            return ElementBasis<ProblemDim, Type>::get_face_expressions();
        }
        else {
            static const std::vector<std::string> empty;
            return empty;
        }
    }

    bool has_parsed_face_expressions() const override
    {
        if constexpr (ElementBasis<ProblemDim, Type>::element_dimension >= 3) {
            auto& parsed = const_cast<ElementBasisWrapper*>(this)
                               ->get_parsed_face_expressions();
            return !parsed.empty();
        }
        return false;
    }

    std::size_t face_expression_count() const override
    {
        if constexpr (ElementBasis<ProblemDim, Type>::element_dimension >= 3) {
            return ElementBasis<ProblemDim, Type>::get_face_expressions()
                .size();
        }
        return 0;
    }

    // Volume expression interface (only if supported)
    void add_volume_expression(const std::string& expr) override
    {
        if constexpr (ElementBasis<ProblemDim, Type>::element_dimension == 3) {
            ElementBasis<ProblemDim, Type>::add_volume_expression(expr);
        }
    }

    const std::vector<std::string>& get_volume_expressions() const override
    {
        if constexpr (ElementBasis<ProblemDim, Type>::element_dimension == 3) {
            return ElementBasis<ProblemDim, Type>::get_volume_expressions();
        }
        else {
            static const std::vector<std::string> empty;
            return empty;
        }
    }

    bool has_parsed_volume_expressions() const override
    {
        if constexpr (ElementBasis<ProblemDim, Type>::element_dimension == 3) {
            auto& parsed = const_cast<ElementBasisWrapper*>(this)
                               ->get_parsed_volume_expressions();
            return !parsed.empty();
        }
        return false;
    }

    std::size_t volume_expression_count() const override
    {
        if constexpr (ElementBasis<ProblemDim, Type>::element_dimension == 3) {
            return ElementBasis<ProblemDim, Type>::get_volume_expressions()
                .size();
        }
        return 0;
    }
};

using ElementBasisHandle = std::shared_ptr<ElementBasisBase>;

// Factory functions for creating ElementBasisHandle
template<unsigned ProblemDim, ElementBasisType Type>
ElementBasisHandle make_element_basis()
{
    return std::make_shared<ElementBasisWrapper<ProblemDim, Type>>();
}

// Convenience factory functions
inline ElementBasisHandle make_fem_1d()
{
    return make_element_basis<1, ElementBasisType::FiniteElement>();
}

inline ElementBasisHandle make_fem_2d()
{
    return make_element_basis<2, ElementBasisType::FiniteElement>();
}

inline ElementBasisHandle make_fem_3d()
{
    return make_element_basis<3, ElementBasisType::FiniteElement>();
}

inline ElementBasisHandle make_bem_2d()
{
    return make_element_basis<2, ElementBasisType::BoundaryElement>();
}

inline ElementBasisHandle make_bem_3d()
{
    return make_element_basis<3, ElementBasisType::BoundaryElement>();
}

}  // namespace fem_bem

USTC_CG_NAMESPACE_CLOSE_SCOPE
