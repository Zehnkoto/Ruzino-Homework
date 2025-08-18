#include <gtest/gtest.h>

#include <iostream>

#include "fem_bem/ElementBasis.hpp"


using namespace USTC_CG::fem_bem;

TEST(DebugCoordinateMappingTest, AnalyzeCoordinateMapping2D)
{
    auto fem2d = make_fem_2d();
    fem2d->add_vertex_expression("u1");

    // Triangle: (0,0), (1,0), (0,1)
    std::vector<pxr::GfVec2d> triangle = { pxr::GfVec2d(0.0, 0.0),
                                           pxr::GfVec2d(1.0, 0.0),
                                           pxr::GfVec2d(0.0, 1.0) };

    std::cout << "=== Debug Coordinate Mapping 2D ===" << std::endl;

    // Test 1: Basic integration without mapping (should work)
    auto results_no_mapping = fem2d->integrate_vertex_against_str("1");
    std::cout << "No mapping - integrate u1 against 1: "
              << results_no_mapping[0] << std::endl;
    EXPECT_NEAR(results_no_mapping[0], 1.0 / 3.0, 1e-3);

    // Test 2: Integration with mapping against constant (should equal no
    // mapping)
    auto results_const =
        fem2d->integrate_vertex_against_with_mapping("1", triangle);
    std::cout << "With mapping - integrate u1 against 1: " << results_const[0]
              << std::endl;
    EXPECT_NEAR(results_const[0], 1.0 / 3.0, 1e-3);

    // Test 3: Integration with mapping against x coordinate
    auto results_x =
        fem2d->integrate_vertex_against_with_mapping("x", triangle);
    std::cout << "With mapping - integrate u1 against x: " << results_x[0]
              << std::endl;

    // Test 4: Integration with mapping against y coordinate
    auto results_y =
        fem2d->integrate_vertex_against_with_mapping("y", triangle);
    std::cout << "With mapping - integrate u1 against y: " << results_y[0]
              << std::endl;

    // Test 5: Manual coordinate mapping test
    std::cout << "\n=== Manual Coordinate Mapping Test ===" << std::endl;

    // Create coordinate mapping manually
    std::vector<std::string> barycentric_names = { "u1", "u2" };
    auto coord_mapping = create_coordinate_mapping(barycentric_names, triangle);

    std::cout << "Created coordinate mapping with " << coord_mapping.size()
              << " coordinates" << std::endl;

    // Test the coordinate expressions directly
    ParameterMap<real> test_coords;
    test_coords.insert_or_assign("u1", 0.5f);
    test_coords.insert_or_assign("u2", 0.3f);

    if (coord_mapping.find("x")) {
        const Expression& x_expr = *coord_mapping.find("x");
        real x_value = x_expr.evaluate_at(test_coords);
        std::cout << "x mapping at (u1=0.5, u2=0.3): " << x_value << std::endl;
        std::cout << "x expression string: " << x_expr.get_string()
                  << std::endl;
    }

    if (coord_mapping.find("y")) {
        const Expression& y_expr = *coord_mapping.find("y");
        real y_value = y_expr.evaluate_at(test_coords);
        std::cout << "y mapping at (u1=0.5, u2=0.3): " << y_value << std::endl;
        std::cout << "y expression string: " << y_expr.get_string()
                  << std::endl;
    }
}

TEST(DebugCoordinateMappingTest, AnalyzeBEM2DMapping)
{
    auto bem2d = make_bem_2d();
    bem2d->add_vertex_expression("u1");
    bem2d->add_vertex_expression("1 - u1");

    // Line segment from (0,0) to (2,0)
    std::vector<pxr::GfVec2d> line = { pxr::GfVec2d(0.0, 0.0),
                                       pxr::GfVec2d(2.0, 0.0) };

    std::cout << "\n=== Debug BEM2D Mapping ===" << std::endl;

    // Test 1: Basic integration without mapping
    auto results_no_mapping = bem2d->integrate_vertex_against_str("1");
    std::cout << "No mapping - vertex expressions count: "
              << results_no_mapping.size() << std::endl;
    for (size_t i = 0; i < results_no_mapping.size(); ++i) {
        std::cout << "  Result[" << i << "]: " << results_no_mapping[i]
                  << std::endl;
    }

    // Test 2: Integration with mapping against constant
    auto results_const =
        bem2d->integrate_vertex_against_with_mapping("1", line);
    std::cout << "With mapping - integrate against 1:" << std::endl;
    for (size_t i = 0; i < results_const.size(); ++i) {
        std::cout << "  Result[" << i << "]: " << results_const[i] << std::endl;
    }

    // Test 3: Integration with mapping against x coordinate
    auto results_x = bem2d->integrate_vertex_against_with_mapping("x", line);
    std::cout << "With mapping - integrate against x:" << std::endl;
    for (size_t i = 0; i < results_x.size(); ++i) {
        std::cout << "  Result[" << i << "]: " << results_x[i] << std::endl;
    }

    // Test 4: Manual coordinate mapping for BEM2D
    std::cout << "\n=== Manual BEM2D Coordinate Mapping Test ===" << std::endl;

    std::vector<std::string> barycentric_names = { "u1" };
    auto coord_mapping = create_coordinate_mapping(barycentric_names, line);

    std::cout << "BEM2D coordinate mapping created with "
              << coord_mapping.size() << " coordinates" << std::endl;

    // Test the coordinate expressions directly
    ParameterMap<real> test_coords;
    test_coords.insert_or_assign("u1", 0.25f);

    if (coord_mapping.find("x")) {
        const Expression& x_expr = *coord_mapping.find("x");
        real x_value = x_expr.evaluate_at(test_coords);
        std::cout << "x mapping at u1=0.25: " << x_value << " (expected: 0.5)"
                  << std::endl;
        std::cout << "x expression string: " << x_expr.get_string()
                  << std::endl;
    }

    if (coord_mapping.find("y")) {
        const Expression& y_expr = *coord_mapping.find("y");
        real y_value = y_expr.evaluate_at(test_coords);
        std::cout << "y mapping at u1=0.25: " << y_value << " (expected: 0.0)"
                  << std::endl;
        std::cout << "y expression string: " << y_expr.get_string()
                  << std::endl;
    }
}

TEST(DebugCoordinateMappingTest, TestMappedExpressionCreation)
{
    std::cout << "\n=== Test Mapped Expression Creation ===" << std::endl;

    // Create a simple expression
    Expression simple_x("x", { "x" });

    // Create coordinate mapping
    std::vector<std::string> barycentric_names = { "u1", "u2" };
    std::vector<pxr::GfVec2d> triangle = { pxr::GfVec2d(0.0, 0.0),
                                           pxr::GfVec2d(1.0, 0.0),
                                           pxr::GfVec2d(0.0, 1.0) };

    auto coord_mapping = create_coordinate_mapping(barycentric_names, triangle);

    // Create mapped expression
    auto mapped_expr = create_mapped_expression_with_coord_mapping(
        simple_x, coord_mapping, barycentric_names);

    std::cout << "Original expression: " << simple_x.get_string() << std::endl;
    std::cout << "Mapped expression: " << mapped_expr.get_string() << std::endl;

    // Test evaluation
    ParameterMap<real> test_coords;
    test_coords.insert_or_assign("u1", 0.5f);
    test_coords.insert_or_assign("u2", 0.3f);

    real result = mapped_expr.evaluate_at(test_coords);
    std::cout << "Mapped expression at (u1=0.5, u2=0.3): " << result
              << std::endl;
    std::cout << "Expected: " << (1.0 - 0.5 - 0.3) * 0.0 + 0.5 * 1.0 + 0.3 * 0.0
              << std::endl;
}

TEST(DebugIntegrationPipeline, TraceIntegrationSteps)
{
    std::cout << "\n=== Debug Integration Pipeline ===" << std::endl;

    auto fem2d = make_fem_2d();
    fem2d->add_vertex_expression("u1");

    // Triangle: (0,0), (1,0), (0,1)
    std::vector<pxr::GfVec2d> triangle = { pxr::GfVec2d(0.0, 0.0),
                                           pxr::GfVec2d(1.0, 0.0),
                                           pxr::GfVec2d(0.0, 1.0) };

    // Step 1: Test basic integration (should work)
    std::cout << "Step 1: Basic integration without mapping" << std::endl;
    auto basic_result = fem2d->integrate_vertex_against_str("1");
    std::cout << "  Result: " << basic_result[0] << " (expected: ~0.333)"
              << std::endl;

    // Step 2: Test coordinate mapping creation
    std::cout << "\nStep 2: Create coordinate mapping" << std::endl;
    std::vector<std::string> barycentric_names = { "u1", "u2" };
    auto coord_mapping = create_coordinate_mapping(barycentric_names, triangle);
    std::cout << "  Created mapping with " << coord_mapping.size()
              << " coordinates" << std::endl;

    // Step 3: Test individual coordinate expressions
    std::cout << "\nStep 3: Test coordinate expressions directly" << std::endl;
    ParameterMap<real> test_coords;
    test_coords.insert_or_assign("u1", 0.5f);
    test_coords.insert_or_assign("u2", 0.3f);

    if (coord_mapping.find("x")) {
        const Expression& x_expr = *coord_mapping.find("x");
        real x_value = x_expr.evaluate_at(test_coords);
        std::cout << "  x at (0.5, 0.3): " << x_value << " (expected: 0.5)"
                  << std::endl;
    }

    // Step 4: Test simple mapped expression
    std::cout << "\nStep 4: Test simple mapped expression creation"
              << std::endl;
    Expression simple_constant("1", barycentric_names);
    auto mapped_constant = create_mapped_expression_with_coord_mapping(
        simple_constant, coord_mapping, barycentric_names);

    real mapped_value = mapped_constant.evaluate_at(test_coords);
    std::cout << "  Mapped constant at (0.5, 0.3): " << mapped_value
              << " (expected: 1.0)" << std::endl;

    // Step 5: Test expression multiplication
    std::cout << "\nStep 5: Test expression multiplication" << std::endl;
    Expression u1_expr("u1", barycentric_names);
    Expression product = u1_expr * mapped_constant;

    real product_value = product.evaluate_at(test_coords);
    std::cout << "  u1 * mapped_constant at (0.5, 0.3): " << product_value
              << " (expected: 0.5)" << std::endl;

    // Step 6: Test manual integration of product
    std::cout << "\nStep 6: Manual integration test" << std::endl;

    // Create the product expression that should be integrated
    Expression integrand = u1_expr * mapped_constant;

    // Test integrand at a few points
    ParameterMap<real> test_point1;
    test_point1.insert_or_assign("u1", 0.1f);
    test_point1.insert_or_assign("u2", 0.1f);
    real val1 = integrand.evaluate_at(test_point1);
    std::cout << "  Integrand at (0.1, 0.1): " << val1 << std::endl;

    ParameterMap<real> test_point2;
    test_point2.insert_or_assign("u1", 0.3f);
    test_point2.insert_or_assign("u2", 0.2f);
    real val2 = integrand.evaluate_at(test_point2);
    std::cout << "  Integrand at (0.3, 0.2): " << val2 << std::endl;

    // Step 7: Try direct integration call
    std::cout << "\nStep 7: Direct integration call" << std::endl;
    try {
        real direct_result = integrate_over_simplex(
            integrand, barycentric_names, coord_mapping, 100);
        std::cout << "  Direct integration result: " << direct_result
                  << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "  Direct integration failed: " << e.what() << std::endl;
    }
}

int main()
{
    ::testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
