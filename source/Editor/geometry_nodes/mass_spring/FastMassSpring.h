#pragma once
#include <Eigen/Sparse>
#include <memory>

#include "MassSpring.h"

namespace USTC_CG::mass_spring {
// Impliment the Liu13's paper:
// https://tiantianliu.cn/papers/liu13fast/liu13fast.pdf
class FastMassSpring : public MassSpring {
   public:
    FastMassSpring() = default;
    ~FastMassSpring() = default;

    FastMassSpring(
        const Eigen::MatrixXd& X,
        const EdgeSet& E,
        const float stiffness,
        const float h);
    void step() override;
    unsigned max_iter =
        10;  // (HW Optional) add UI for this parameter. 10 is recommended by
             // the paper for real-time simulation.

   protected:
    // Custom variables, like prefactorized A
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>
        solver;  // Pre-factorized Cholesky solver
};
}  // namespace USTC_CG::mass_spring