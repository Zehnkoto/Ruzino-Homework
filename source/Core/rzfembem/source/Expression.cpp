#include <cstring>
#include <fem_bem/Expression.hpp>

namespace USTC_CG {
namespace fem_bem {

    real numerical_derivative(
        const exprtk::expression<real>& expr,
        exprtk::details::variable_node<real>* var,
        const real& h)
    {
        const real x_init = var->ref();

        // Use 2-point central difference with better stability
        var->ref() = x_init + h;
        const real y_plus = expr.value();
        var->ref() = x_init - h;
        const real y_minus = expr.value();
        var->ref() = x_init;

        // Check for numerical issues and use adaptive step if needed
        real derivative = (y_plus - y_minus) / (real(2) * h);

        // If derivative seems to large, try with smaller step
        if (std::abs(derivative) > real(1e6)) {
            real smaller_h = h * real(0.1);
            var->ref() = x_init + smaller_h;
            const real y_plus_small = expr.value();
            var->ref() = x_init - smaller_h;
            const real y_minus_small = expr.value();
            var->ref() = x_init;
            derivative = (y_plus_small - y_minus_small) / (real(2) * smaller_h);
        }

        return derivative;
    }
    std::function<real(const ParameterMap<real>&)> create_derivative_function(
        const std::string& expression_string,
        const std::string& variable_name,
        const real& h)
    {
        return [expression_string, variable_name, h](
                   const ParameterMap<real>& values) -> real {
            // Create a temporary expression for derivative computation
            exprtk::symbol_table<real> symbol_table;
            exprtk::expression<real> expr;
            exprtk::parser<real> parser;

            // Create a mutable copy of values for symbol table
            ParameterMap<real> temp_values = values;

            // Ensure all coordinate variables exist with default values
            if (!temp_values.find("x")) {
                temp_values.insert_or_assign("x", real(0));
            }
            if (!temp_values.find("y")) {
                temp_values.insert_or_assign("y", real(0));
            }
            if (!temp_values.find("z")) {
                temp_values.insert_or_assign("z", real(0));
            }

            // Ensure barycentric coordinates exist with default values
            if (!temp_values.find("u1")) {
                temp_values.insert_or_assign("u1", real(0));
            }
            if (!temp_values.find("u2")) {
                temp_values.insert_or_assign("u2", real(0));
            }
            if (!temp_values.find("u3")) {
                temp_values.insert_or_assign("u3", real(0));
            }

            // Add all variables to symbol table using mutable references
            for (std::size_t i = 0; i < temp_values.size(); ++i) {
                const char* name = temp_values.get_name_at(i);
                real& value_ref =
                    const_cast<real&>(temp_values.get_value_at(i));
                symbol_table.add_variable(name, value_ref);
            }

            symbol_table.add_constants();
            expr.register_symbol_table(symbol_table);

            if (!parser.compile(expression_string, expr)) {
                return real(0);  // Return 0 for invalid expressions
            }

            // Find the variable node for differentiation
            auto* var_node = symbol_table.get_variable(variable_name);
            if (!var_node) {
                return real(0);  // Variable not found
            }

            return numerical_derivative(expr, var_node, h);
        };
    }

    // Create a derivative function for compound expressions using chain rule
    std::function<real(const ParameterMap<real>&)>
    create_compound_derivative_function(
        const std::function<real(const ParameterMap<real>&)>&
            compound_evaluator,
        const std::string& variable_name,
        const real& h)
    {
        return [compound_evaluator, variable_name, h](
                   const ParameterMap<real>& values) -> real {
            const real* var_value = values.find(variable_name.c_str());
            if (!var_value) {
                return real(0);  // Variable not found
            }

            const real x_init = *var_value;

            // Create modified value maps for derivative computation
            ParameterMap<real> values_plus = values;
            ParameterMap<real> values_minus = values;

            // Use 2-point central difference
            values_plus.insert_or_assign(variable_name.c_str(), x_init + h);
            const real y_plus = compound_evaluator(values_plus);

            values_minus.insert_or_assign(variable_name.c_str(), x_init - h);
            const real y_minus = compound_evaluator(values_minus);

            real derivative = (y_plus - y_minus) / (real(2) * h);

            // Check for numerical issues and use adaptive step if needed
            if (std::abs(derivative) > real(1e6)) {
                real smaller_h = h * real(0.1);
                values_plus.insert_or_assign(
                    variable_name.c_str(), x_init + smaller_h);
                const real y_plus_small = compound_evaluator(values_plus);

                values_minus.insert_or_assign(
                    variable_name.c_str(), x_init - smaller_h);
                const real y_minus_small = compound_evaluator(values_minus);

                derivative =
                    (y_plus_small - y_minus_small) / (real(2) * smaller_h);
            }

            return derivative;
        };
    }

    Expression::Expression(const std::string& expr_str)
        : expression_string_(expr_str),
          is_compound_(false)
    {
    }
    Expression::Expression(
        const std::string& expr_str,
        const std::vector<std::string>& variable_names)
        : expression_string_(expr_str),
          variable_names_(variable_names),
          is_compound_(false)
    {
    }

    Expression::Expression(
        const Expression& outer_expr,
        const std::vector<std::pair<const char*, Expression>>&
            variable_substitutions)
        : outer_expression_(std::make_unique<Expression>(outer_expr)),
          substitution_map_(variable_substitutions),
          is_compound_(true)
    {
        // Build compound expression string for display
        expression_string_ = outer_expr.get_string();

        // Apply recursive substitution logic
        apply_recursive_substitution();
    }
    Expression::Expression(
        const Expression& outer_expr,
        std::initializer_list<std::pair<const char*, Expression>> substitutions)
        : outer_expression_(std::make_unique<Expression>(outer_expr)),
          substitution_map_(substitutions.begin(), substitutions.end()),
          is_compound_(true)
    {
        // Build compound expression string for display
        expression_string_ = outer_expr.get_string();

        // Apply recursive substitution logic
        apply_recursive_substitution();
    }

    Expression::Expression(const Expression& other)
        : expression_string_(other.expression_string_),
          variable_names_(other.variable_names_),
          is_parsed_(false),  // Force re-parsing with new symbol table
          is_compound_(other.is_compound_),
          outer_expression_(
              other.outer_expression_
                  ? std::make_unique<Expression>(*other.outer_expression_)
                  : nullptr),
          substitution_map_(other.substitution_map_),
          derivative_evaluator_(other.derivative_evaluator_),
          bound_variables_(other.bound_variables_)
    {
    }
    Expression& Expression::operator=(const Expression& other)
    {
        if (this != &other) {
            expression_string_ = other.expression_string_;
            variable_names_ = other.variable_names_;
            is_parsed_ = false;  // Force re-parsing
            compiled_expression_.reset();
            symbol_table_.reset();
            is_compound_ = other.is_compound_;
            outer_expression_ =
                other.outer_expression_
                    ? std::make_unique<Expression>(*other.outer_expression_)
                    : nullptr;
            substitution_map_ = other.substitution_map_;
            derivative_evaluator_ = other.derivative_evaluator_;
            bound_variables_ = other.bound_variables_;
        }
        return *this;
    }
    Expression Expression::from_string(const std::string& expr_str)
    {
        return Expression(expr_str);
    }
    Expression Expression::constant(real value)
    {
        return Expression(std::to_string(value));
    }
    Expression Expression::zero()
    {
        return Expression("0");
    }
    Expression Expression::one()
    {
        return Expression("1");
    }
    const std::string& Expression::get_string() const
    {
        return expression_string_;
    }
    bool Expression::is_string_based() const
    {
        return true;  // Regular expressions are always string-based
    }
    Expression Expression::bind_variables(
        const ParameterMap<real>& bound_values) const
    {
        Expression closure = *this;

        // Merge bound values with existing ones
        for (std::size_t i = 0; i < bound_values.size(); ++i) {
            const char* name = bound_values.get_name_at(i);
            const real& value = bound_values.get_value_at(i);
            closure.bound_variables_.insert_or_assign(name, value);
        }

        return closure;
    }

    Expression Expression::bind_variable(
        const std::string& var_name,
        real value) const
    {
        Expression closure = *this;
        closure.bound_variables_.insert_or_assign(var_name.c_str(), value);
        return closure;
    }
    bool Expression::has_bound_variables() const
    {
        return !bound_variables_.empty();
    }
    const ParameterMap<real>& Expression::get_bound_variables() const
    {
        return bound_variables_;
    }
    const Expression::expression_type* Expression::get_compiled_expression()
        const
    {
        ensure_parsed();
        return compiled_expression_.get();
    }
    const Expression::symbol_table_type* Expression::get_symbol_table() const
    {
        ensure_parsed();
        return symbol_table_.get();
    }

    real Expression::evaluate_at(
        const ParameterMap<real>& variable_values) const
    {
        // Handle expressions created from DerivativeExpression
        if (derivative_evaluator_) {
            // Merge bound variables with provided values
            ParameterMap<real> merged_values = bound_variables_;
            for (std::size_t i = 0; i < variable_values.size(); ++i) {
                const char* name = variable_values.get_name_at(i);
                const real& value = variable_values.get_value_at(i);
                merged_values.insert_or_assign(name, value);
            }
            return derivative_evaluator_(merged_values);
        }

        // Handle compound expressions
        if (is_compound_ && outer_expression_) {
            // Merge bound variables with provided values for substitution
            // evaluation
            ParameterMap<real> merged_values = bound_variables_;
            for (std::size_t i = 0; i < variable_values.size(); ++i) {
                const char* name = variable_values.get_name_at(i);
                const real& value = variable_values.get_value_at(i);
                merged_values.insert_or_assign(name, value);
            }

            // Evaluate substitutions
            ParameterMap<real> outer_values;
            for (std::size_t i = 0; i < substitution_map_.size(); ++i) {
                const auto& pair = substitution_map_[i];
                real sub_result = pair.second.evaluate_at(merged_values);
                outer_values.insert_or_assign(pair.first, sub_result);
            }

            return outer_expression_->evaluate_at(outer_values);
        }

        // Standard evaluation for non-compound expressions
        if (!is_parsed_ || !compiled_expression_) {
            // If no variables specified, discover them from the values provided
            if (variable_names_.empty()) {
                for (std::size_t i = 0; i < variable_values.size(); ++i) {
                    variable_names_.push_back(variable_values.get_name_at(i));
                }
                // Also add bound variable names
                for (std::size_t i = 0; i < bound_variables_.size(); ++i) {
                    variable_names_.push_back(bound_variables_.get_name_at(i));
                }
            }
            parse_expression();
        }

        if (!compiled_expression_) {
            throw std::runtime_error(
                "Expression not properly parsed: " + expression_string_);
        }

        // Merge bound variables with provided values (provided values take
        // precedence)
        ParameterMap<real> merged_values = bound_variables_;
        for (std::size_t i = 0; i < variable_values.size(); ++i) {
            const char* name = variable_values.get_name_at(i);
            const real& value = variable_values.get_value_at(i);
            merged_values.insert_or_assign(name, value);
        }

        // Set all variables in temp_variables_
        for (std::size_t i = 0; i < merged_values.size(); ++i) {
            const char* name = merged_values.get_name_at(i);
            const real& value = merged_values.get_value_at(i);
            auto ptr = temp_variables_.find(name);
            if (ptr)
                *ptr = value;
        }

        real result = compiled_expression_->value();
        return result;
    }

    Expression Expression::operator+(const Expression& other) const
    {
        Expression add_expr("xx_ + yy_", { "xx_", "yy_" });
        return Expression(add_expr, { { "xx_", *this }, { "yy_", other } });
    }

    Expression Expression::operator-(const Expression& other) const
    {
        Expression sub_expr("xx_ - yy_", { "xx_", "yy_" });
        return Expression(sub_expr, { { "xx_", *this }, { "yy_", other } });
    }

    Expression Expression::operator*(const Expression& other) const
    {
        Expression mul_expr("xx_ * yy_", { "xx_", "yy_" });
        return Expression(mul_expr, { { "xx_", *this }, { "yy_", other } });
    }

    Expression Expression::operator/(const Expression& other) const
    {
        Expression div_expr("xx_ / yy_", { "xx_", "yy_" });
        return Expression(div_expr, { { "xx_", *this }, { "yy_", other } });
    }

    Expression Expression::operator*(real scalar) const
    {
        Expression scalar_expr(std::to_string(scalar));
        Expression mul_expr("xx_ * yy_", { "xx_", "yy_" });
        return Expression(
            mul_expr, { { "xx_", scalar_expr }, { "yy_", *this } });
    }

    Expression Expression::operator-() const
    {
        Expression neg_expr("-xx_", { "xx_" });
        return Expression(neg_expr, { { "xx_", *this } });
    }

    DerivativeExpression Expression::derivative(
        const std::string& variable_name) const
    {
        // Use more conservative step sizes for float precision
        real h;
        if (is_compound_ && outer_expression_) {
            // Check if any of the substitutions are derivatives
            bool has_derivative_substitution = false;
            for (const auto& pair : substitution_map_) {
                if (pair.second.derivative_evaluator_) {
                    has_derivative_substitution = true;
                    break;
                }
            }
            // For compound expressions with derivatives, use significantly
            // larger step
            h = has_derivative_substitution ? real(5e-3) : real(1e-4);
        }
        else if (derivative_evaluator_) {
            // This is already a derivative, so we're computing second
            // derivative
            h = real(5e-3);
        }
        else {
            // Simple expression
            h = real(1e-4);
        }

        // For compound expressions, use numerical chain rule
        if (is_compound_ && outer_expression_) {
            auto compound_evaluator = [this](const ParameterMap<real>& values) {
                return this->evaluate_at(values);
            };
            auto derivative_func = create_compound_derivative_function(
                compound_evaluator, variable_name, h);
            return DerivativeExpression(derivative_func, variable_name);
        }
        // Handle derivative expressions (derivatives of derivatives)
        else if (derivative_evaluator_) {
            auto derivative_func = create_compound_derivative_function(
                derivative_evaluator_, variable_name, h);
            return DerivativeExpression(derivative_func, variable_name);
        }
        else {
            // For simple expressions, use string-based derivative
            auto derivative_func = create_derivative_function(
                expression_string_, variable_name, h);
            return DerivativeExpression(derivative_func, variable_name);
        }
    }
    void Expression::ensure_parsed() const
    {
        if (!is_parsed_ || !compiled_expression_) {
            parse_expression();
        }
    }
    void Expression::parse_expression() const
    {
        symbol_table_ = std::make_unique<symbol_table_type>();
        compiled_expression_ = std::make_unique<expression_type>();

        // Enable constants and standard functions
        symbol_table_->add_constants();

        // Register specified variables only
        if (!variable_names_.empty()) {
            for (const auto& var_name : variable_names_) {
                auto* temp_var_ptr = temp_variables_.find(var_name.c_str());
                if (!temp_var_ptr) {
                    temp_variables_.insert_or_assign(var_name.c_str(), real(0));
                    temp_var_ptr = temp_variables_.find(var_name.c_str());
                }
                symbol_table_->add_variable(
                    var_name, const_cast<real&>(*temp_var_ptr));
            }
        }

        // Register symbol table with expression
        compiled_expression_->register_symbol_table(*symbol_table_);

        parser_type parser;
        if (!parser.compile(expression_string_, *compiled_expression_)) {
            throw std::runtime_error(
                "Failed to parse expression: " + expression_string_);
        }

        is_parsed_ = true;
    }

    void Expression::apply_recursive_substitution()
    {
        // If the outer expression is not compound, nothing special to do
        if (!outer_expression_ || !outer_expression_->is_compound_) {
            return;
        }

        // This method implements the recursive substitution logic you
        // described:
        // 1. Variables that are re-substituted don't get double-substituted
        // 2. New substitutions are applied to existing expressions recursively
        // Example: if outer has {xx: u1+u2} and new has {xx: u2, u2: u3}
        // Result should be {u2: u3, yy: whatever_yy_was} (xx is gone, u2
        // becomes u3)

        std::vector<std::pair<const char*, Expression>> final_substitutions;
        const auto& outer_substitutions = outer_expression_->substitution_map_;

        // Process each variable from the outer expression's substitution map
        for (const auto& outer_pair : outer_substitutions) {
            const char* var_name = outer_pair.first;
            Expression outer_expr =
                outer_pair.second;  // Make a copy for modification

            // Check if this variable is being re-substituted in our new
            // substitution map
            bool is_being_resubstituted = false;
            for (const auto& new_pair : substitution_map_) {
                if (std::strcmp(new_pair.first, var_name) == 0) {
                    // This variable is being re-substituted, so we skip the
                    // outer substitution (completed substitutions shouldn't be
                    // re-substituted)
                    is_being_resubstituted = true;
                    break;
                }
            }

            if (!is_being_resubstituted) {
                // Apply any relevant new substitutions to this outer expression
                // recursively
                for (const auto& new_pair : substitution_map_) {
                    const char* new_var = new_pair.first;
                    const Expression& new_expr = new_pair.second;

                    // Check if the outer expression contains the variable we're
                    // substituting
                    if (outer_expr.is_compound_) {
                        // For compound expressions, check if any substitution
                        // uses this variable
                        bool contains_var = false;
                        for (const auto& sub_pair :
                             outer_expr.substitution_map_) {
                            if (std::strcmp(sub_pair.first, new_var) == 0) {
                                contains_var = true;
                                break;
                            }
                        }
                        if (contains_var) {
                            // Apply the substitution recursively
                            outer_expr = Expression(
                                outer_expr, { { new_var, new_expr } });
                        }
                    }
                    else {
                        // For simple expressions, check if the variable appears
                        // in the string
                        if (outer_expr.expression_string_.find(new_var) !=
                            std::string::npos) {
                            // Create a compound expression with the
                            // substitution
                            outer_expr = Expression(
                                outer_expr, { { new_var, new_expr } });
                        }
                    }
                }

                // Add the (possibly recursively modified) outer substitution
                final_substitutions.emplace_back(var_name, outer_expr);
            }
        }

        // Add our new substitutions
        for (const auto& new_pair : substitution_map_) {
            final_substitutions.emplace_back(new_pair.first, new_pair.second);
        }

        // Update our substitution map with the final results
        substitution_map_ = std::move(final_substitutions);

        // Update the outer expression to be the base expression (without its
        // substitutions) This ensures we don't double-apply the outer
        // substitutions
        if (outer_expression_->outer_expression_) {
            outer_expression_ = std::make_unique<Expression>(
                *outer_expression_->outer_expression_);
        }
        else {
            // Create a simple expression with just the string
            Expression simple_expr(outer_expression_->expression_string_);
            outer_expression_ = std::make_unique<Expression>(simple_expr);
            outer_expression_->is_compound_ = false;
        }
    }

    Expression operator*(real scalar, const Expression& expr)
    {
        return expr * scalar;
    }
    Expression make_expression(const std::string& expr_str)
    {
        return Expression::from_string(expr_str);
    }

    std::vector<DerivativeExpression> Expression::gradient(
        const std::vector<std::string>& variable_names) const
    {
        std::vector<DerivativeExpression> grad;
        grad.reserve(variable_names.size());
        for (const auto& var : variable_names) {
            grad.push_back(derivative(var));
        }
        return grad;
    }

    // Create coordinate mapping expressions that bind world vertex coordinates
    ParameterMap<Expression> create_coordinate_mapping(
        const std::vector<std::string>& barycentric_names,
        const std::vector<pxr::GfVec2d>& world_vertices)
    {
        ParameterMap<Expression> coord_mapping;

        // Handle edge cases gracefully
        if (world_vertices.empty()) {
            coord_mapping.insert_or_assign("x", Expression("0"));
            coord_mapping.insert_or_assign("y", Expression("0"));
            return coord_mapping;
        }

        if (world_vertices.size() == 1) {
            // Single point case
            coord_mapping.insert_or_assign(
                "x", Expression::constant(world_vertices[0][0]));
            coord_mapping.insert_or_assign(
                "y", Expression::constant(world_vertices[0][1]));
            return coord_mapping;
        }

        // For BEM2D (1D elements): we have 2 vertices and 1 barycentric
        // coordinate u1 The mapping is: x = (1-u1)*x0 + u1*x1, y = (1-u1)*y0 +
        // u1*y1
        if (barycentric_names.size() == 1 && world_vertices.size() >= 2) {
            // Create x mapping: (1-u1)*x0 + u1*x1
            std::vector<std::string> x_vars = { "u1", "x0", "x1" };
            Expression x_expr("(1 - u1) * x0 + u1 * x1", x_vars);
            ParameterMap<real> x_bindings;
            x_bindings.insert_or_assign(
                "x0", static_cast<real>(world_vertices[0][0]));
            x_bindings.insert_or_assign(
                "x1", static_cast<real>(world_vertices[1][0]));
            x_expr = x_expr.bind_variables(x_bindings);

            // Create y mapping: (1-u1)*y0 + u1*y1
            std::vector<std::string> y_vars = { "u1", "y0", "y1" };
            Expression y_expr("(1 - u1) * y0 + u1 * y1", y_vars);
            ParameterMap<real> y_bindings;
            y_bindings.insert_or_assign(
                "y0", static_cast<real>(world_vertices[0][1]));
            y_bindings.insert_or_assign(
                "y1", static_cast<real>(world_vertices[1][1]));
            y_expr = y_expr.bind_variables(y_bindings);

            coord_mapping.insert_or_assign("x", x_expr);
            coord_mapping.insert_or_assign("y", y_expr);
            return coord_mapping;
        }

        // For FEM2D (2D elements): we have 3 vertices and 2 barycentric
        // coordinates u1, u2 The mapping is: x = (1-u1-u2)*x0 + u1*x1 + u2*x2
        if (barycentric_names.size() == 2 && world_vertices.size() >= 3) {
            // Create x mapping: (1-u1-u2)*x0 + u1*x1 + u2*x2
            std::vector<std::string> x_vars = { "u1", "u2", "x0", "x1", "x2" };
            Expression x_expr("(1 - u1 - u2) * x0 + u1 * x1 + u2 * x2", x_vars);
            ParameterMap<real> x_bindings;
            x_bindings.insert_or_assign(
                "x0", static_cast<real>(world_vertices[0][0]));
            x_bindings.insert_or_assign(
                "x1", static_cast<real>(world_vertices[1][0]));
            x_bindings.insert_or_assign(
                "x2", static_cast<real>(world_vertices[2][0]));
            x_expr = x_expr.bind_variables(x_bindings);

            // Create y mapping: (1-u1-u2)*y0 + u1*y1 + u2*y2
            std::vector<std::string> y_vars = { "u1", "u2", "y0", "y1", "y2" };
            Expression y_expr("(1 - u1 - u2) * y0 + u1 * y1 + u2 * y2", y_vars);
            ParameterMap<real> y_bindings;
            y_bindings.insert_or_assign(
                "y0", static_cast<real>(world_vertices[0][1]));
            y_bindings.insert_or_assign(
                "y1", static_cast<real>(world_vertices[1][1]));
            y_bindings.insert_or_assign(
                "y2", static_cast<real>(world_vertices[2][1]));
            y_expr = y_expr.bind_variables(y_bindings);

            coord_mapping.insert_or_assign("x", x_expr);
            coord_mapping.insert_or_assign("y", y_expr);
            return coord_mapping;
        }

        // Fallback for other cases
        std::size_t actual_vertices =
            std::min(world_vertices.size(), barycentric_names.size() + 1);

        // Build the base expression string for x coordinate
        std::string x_base = "(1";
        for (std::size_t i = 0; i < barycentric_names.size(); ++i) {
            x_base += " - " + barycentric_names[i];
        }
        x_base += ") * x0";

        for (std::size_t i = 0;
             i < std::min(barycentric_names.size(), actual_vertices - 1);
             ++i) {
            x_base +=
                " + " + barycentric_names[i] + " * x" + std::to_string(i + 1);
        }

        // Create variable list including all coordinate variables
        std::vector<std::string> x_vars = barycentric_names;
        for (std::size_t i = 0; i < actual_vertices; ++i) {
            x_vars.push_back("x" + std::to_string(i));
        }

        Expression x_expr(x_base, x_vars);
        ParameterMap<real> x_bindings;
        for (std::size_t i = 0; i < actual_vertices; ++i) {
            x_bindings.insert_or_assign(
                ("x" + std::to_string(i)).c_str(),
                static_cast<real>(world_vertices[i][0]));
        }
        x_expr = x_expr.bind_variables(x_bindings);

        // Build the base expression string for y coordinate
        std::string y_base = "(1";
        for (std::size_t i = 0; i < barycentric_names.size(); ++i) {
            y_base += " - " + barycentric_names[i];
        }
        y_base += ") * y0";

        for (std::size_t i = 0;
             i < std::min(barycentric_names.size(), actual_vertices - 1);
             ++i) {
            y_base +=
                " + " + barycentric_names[i] + " * y" + std::to_string(i + 1);
        }

        // Create variable list including all coordinate variables
        std::vector<std::string> y_vars = barycentric_names;
        for (std::size_t i = 0; i < actual_vertices; ++i) {
            y_vars.push_back("y" + std::to_string(i));
        }

        Expression y_expr(y_base, y_vars);
        ParameterMap<real> y_bindings;
        for (std::size_t i = 0; i < actual_vertices; ++i) {
            y_bindings.insert_or_assign(
                ("y" + std::to_string(i)).c_str(),
                static_cast<real>(world_vertices[i][1]));
        }
        y_expr = y_expr.bind_variables(y_bindings);

        coord_mapping.insert_or_assign("x", x_expr);
        coord_mapping.insert_or_assign("y", y_expr);

        return coord_mapping;
    }

    ParameterMap<Expression> create_coordinate_mapping(
        const std::vector<std::string>& barycentric_names,
        const std::vector<pxr::GfVec3d>& world_vertices)
    {
        ParameterMap<Expression> coord_mapping;

        // Handle edge cases gracefully
        if (world_vertices.empty()) {
            coord_mapping.insert_or_assign("x", Expression("0"));
            coord_mapping.insert_or_assign("y", Expression("0"));
            coord_mapping.insert_or_assign("z", Expression("0"));
            return coord_mapping;
        }

        if (world_vertices.size() == 1) {
            coord_mapping.insert_or_assign(
                "x", Expression::constant(world_vertices[0][0]));
            coord_mapping.insert_or_assign(
                "y", Expression::constant(world_vertices[0][1]));
            coord_mapping.insert_or_assign(
                "z", Expression::constant(world_vertices[0][2]));
            return coord_mapping;
        }

        // For BEM2D in 3D (1D elements): we have 2 vertices and 1 barycentric
        // coordinate u1
        if (barycentric_names.size() == 1 && world_vertices.size() >= 2) {
            // Create x mapping: (1-u1)*x0 + u1*x1
            std::vector<std::string> x_vars = { "u1", "x0", "x1" };
            Expression x_expr("(1 - u1) * x0 + u1 * x1", x_vars);
            ParameterMap<real> x_bindings;
            x_bindings.insert_or_assign(
                "x0", static_cast<real>(world_vertices[0][0]));
            x_bindings.insert_or_assign(
                "x1", static_cast<real>(world_vertices[1][0]));
            x_expr = x_expr.bind_variables(x_bindings);

            // Create y mapping
            std::vector<std::string> y_vars = { "u1", "y0", "y1" };
            Expression y_expr("(1 - u1) * y0 + u1 * y1", y_vars);
            ParameterMap<real> y_bindings;
            y_bindings.insert_or_assign(
                "y0", static_cast<real>(world_vertices[0][1]));
            y_bindings.insert_or_assign(
                "y1", static_cast<real>(world_vertices[1][1]));
            y_expr = y_expr.bind_variables(y_bindings);

            // Create z mapping
            std::vector<std::string> z_vars = { "u1", "z0", "z1" };
            Expression z_expr("(1 - u1) * z0 + u1 * z1", z_vars);
            ParameterMap<real> z_bindings;
            z_bindings.insert_or_assign(
                "z0", static_cast<real>(world_vertices[0][2]));
            z_bindings.insert_or_assign(
                "z1", static_cast<real>(world_vertices[1][2]));
            z_expr = z_expr.bind_variables(z_bindings);

            coord_mapping.insert_or_assign("x", x_expr);
            coord_mapping.insert_or_assign("y", y_expr);
            coord_mapping.insert_or_assign("z", z_expr);
            return coord_mapping;
        }

        // For BEM3D (2D elements): we have 3 vertices and 2 barycentric
        // coordinates u1, u2
        if (barycentric_names.size() == 2 && world_vertices.size() >= 3) {
            // Create x mapping: (1-u1-u2)*x0 + u1*x1 + u2*x2
            std::vector<std::string> x_vars = { "u1", "u2", "x0", "x1", "x2" };
            Expression x_expr("(1 - u1 - u2) * x0 + u1 * x1 + u2 * x2", x_vars);
            ParameterMap<real> x_bindings;
            x_bindings.insert_or_assign(
                "x0", static_cast<real>(world_vertices[0][0]));
            x_bindings.insert_or_assign(
                "x1", static_cast<real>(world_vertices[1][0]));
            x_bindings.insert_or_assign(
                "x2", static_cast<real>(world_vertices[2][0]));
            x_expr = x_expr.bind_variables(x_bindings);

            // Create y mapping
            std::vector<std::string> y_vars = { "u1", "u2", "y0", "y1", "y2" };
            Expression y_expr("(1 - u1 - u2) * y0 + u1 * y1 + u2 * y2", y_vars);
            ParameterMap<real> y_bindings;
            y_bindings.insert_or_assign(
                "y0", static_cast<real>(world_vertices[0][1]));
            y_bindings.insert_or_assign(
                "y1", static_cast<real>(world_vertices[1][1]));
            y_bindings.insert_or_assign(
                "y2", static_cast<real>(world_vertices[2][1]));
            y_expr = y_expr.bind_variables(y_bindings);

            // Create z mapping
            std::vector<std::string> z_vars = { "u1", "u2", "z0", "z1", "z2" };
            Expression z_expr("(1 - u1 - u2) * z0 + u1 * z1 + u2 * z2", z_vars);
            ParameterMap<real> z_bindings;
            z_bindings.insert_or_assign(
                "z0", static_cast<real>(world_vertices[0][2]));
            z_bindings.insert_or_assign(
                "z1", static_cast<real>(world_vertices[1][2]));
            z_bindings.insert_or_assign(
                "z2", static_cast<real>(world_vertices[2][2]));
            z_expr = z_expr.bind_variables(z_bindings);

            coord_mapping.insert_or_assign("x", x_expr);
            coord_mapping.insert_or_assign("y", y_expr);
            coord_mapping.insert_or_assign("z", z_expr);
            return coord_mapping;
        }

        // For FEM3D (3D elements): we have 4 vertices and 3 barycentric
        // coordinates u1, u2, u3
        if (barycentric_names.size() == 3 && world_vertices.size() >= 4) {
            // Create x mapping: (1-u1-u2-u3)*x0 + u1*x1 + u2*x2 + u3*x3
            std::vector<std::string> x_vars = { "u1", "u2", "u3", "x0",
                                                "x1", "x2", "x3" };
            Expression x_expr(
                "(1 - u1 - u2 - u3) * x0 + u1 * x1 + u2 * x2 + u3 * x3",
                x_vars);
            ParameterMap<real> x_bindings;
            x_bindings.insert_or_assign(
                "x0", static_cast<real>(world_vertices[0][0]));
            x_bindings.insert_or_assign(
                "x1", static_cast<real>(world_vertices[1][0]));
            x_bindings.insert_or_assign(
                "x2", static_cast<real>(world_vertices[2][0]));
            x_bindings.insert_or_assign(
                "x3", static_cast<real>(world_vertices[3][0]));
            x_expr = x_expr.bind_variables(x_bindings);

            // Similar for y and z
            std::vector<std::string> y_vars = { "u1", "u2", "u3", "y0",
                                                "y1", "y2", "y3" };
            Expression y_expr(
                "(1 - u1 - u2 - u3) * y0 + u1 * y1 + u2 * y2 + u3 * y3",
                y_vars);
            ParameterMap<real> y_bindings;
            y_bindings.insert_or_assign(
                "y0", static_cast<real>(world_vertices[0][1]));
            y_bindings.insert_or_assign(
                "y1", static_cast<real>(world_vertices[1][1]));
            y_bindings.insert_or_assign(
                "y2", static_cast<real>(world_vertices[2][1]));
            y_bindings.insert_or_assign(
                "y3", static_cast<real>(world_vertices[3][1]));
            y_expr = y_expr.bind_variables(y_bindings);

            std::vector<std::string> z_vars = { "u1", "u2", "u3", "z0",
                                                "z1", "z2", "z3" };
            Expression z_expr(
                "(1 - u1 - u2 - u3) * z0 + u1 * z1 + u2 * z2 + u3 * z3",
                z_vars);
            ParameterMap<real> z_bindings;
            z_bindings.insert_or_assign(
                "z0", static_cast<real>(world_vertices[0][2]));
            z_bindings.insert_or_assign(
                "z1", static_cast<real>(world_vertices[1][2]));
            z_bindings.insert_or_assign(
                "z2", static_cast<real>(world_vertices[2][2]));
            z_bindings.insert_or_assign(
                "z3", static_cast<real>(world_vertices[3][2]));
            z_expr = z_expr.bind_variables(z_bindings);

            coord_mapping.insert_or_assign("x", x_expr);
            coord_mapping.insert_or_assign("y", y_expr);
            coord_mapping.insert_or_assign("z", z_expr);
            return coord_mapping;
        }

        // Fallback for other cases
        std::size_t actual_vertices =
            std::min(world_vertices.size(), barycentric_names.size() + 1);

        // Build expressions for x, y, z coordinates
        for (std::size_t coord = 0; coord < 3; ++coord) {
            std::string coord_name = (coord == 0)   ? "x"
                                     : (coord == 1) ? "y"
                                                    : "z";
            std::string var_prefix = coord_name;

            // Build the base expression string
            std::string expr_str = "(1";
            for (std::size_t i = 0; i < barycentric_names.size(); ++i) {
                expr_str += " - " + barycentric_names[i];
            }
            expr_str += ") * " + var_prefix + "0";

            for (std::size_t i = 0;
                 i < std::min(barycentric_names.size(), actual_vertices - 1);
                 ++i) {
                expr_str += " + " + barycentric_names[i] + " * " + var_prefix +
                            std::to_string(i + 1);
            }

            // Create variable list including all coordinate variables
            std::vector<std::string> coord_vars = barycentric_names;
            for (std::size_t i = 0; i < actual_vertices; ++i) {
                coord_vars.push_back(var_prefix + std::to_string(i));
            }

            Expression coord_expr(expr_str, coord_vars);
            ParameterMap<real> coord_bindings;
            for (std::size_t i = 0; i < actual_vertices; ++i) {
                real coord_value =
                    (coord == 0)   ? static_cast<real>(world_vertices[i][0])
                    : (coord == 1) ? static_cast<real>(world_vertices[i][1])
                                   : static_cast<real>(world_vertices[i][2]);
                coord_bindings.insert_or_assign(
                    (var_prefix + std::to_string(i)).c_str(), coord_value);
            }
            coord_expr = coord_expr.bind_variables(coord_bindings);

            coord_mapping.insert_or_assign(coord_name.c_str(), coord_expr);
        }

        return coord_mapping;
    }

    // Helper to create mapped expression using coordinate mapping
    Expression create_mapped_expression_with_coord_mapping(
        const Expression& expr,
        const ParameterMap<Expression>& coord_mapping,
        const std::vector<std::string>& barycentric_names)
    {
        // Create substitution map from coordinate mapping
        std::vector<std::pair<const char*, Expression>> substitutions;

        for (std::size_t i = 0; i < coord_mapping.size(); ++i) {
            const char* coord_name = coord_mapping.get_name_at(i);
            const Expression& coord_expr = coord_mapping.get_value_at(i);
            substitutions.emplace_back(coord_name, coord_expr);
        }

        // Create compound expression that substitutes physical coordinates
        return Expression(expr, substitutions);
    }

}  // namespace fem_bem
}  // namespace USTC_CG
