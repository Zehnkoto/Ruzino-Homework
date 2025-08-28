USTC_CG_ADD_LIB(
	rzpython
	SHARED
	PUBLIC_LIBS GUI nanobind
	PRIVATE_LIBS 
		glm::glm spdlog::spdlog rzconsole Python3::Module
	COMPILE_DEFS
		NOMINMAX=1
)

target_include_directories(rzpython PUBLIC ${NB_DIR}/include)

target_link_libraries(rzpython_runtime_test PUBLIC RHI usd)
target_link_libraries(rzpython_performance_test PUBLIC RHI usd)
target_link_libraries(vtarray_profile_test PUBLIC RHI usd)