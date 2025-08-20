#include "fem_bem/parameter_map.hpp"

namespace USTC_CG {
namespace fem_bem {

    // Global counters for profiling
    std::atomic<std::size_t> g_insert_or_assign_calls{0};
    std::atomic<std::size_t> g_insert_unchecked_calls{0};
    std::atomic<std::size_t> g_evaluate_calls{0};

}  // namespace fem_bem
}  // namespace USTC_CG