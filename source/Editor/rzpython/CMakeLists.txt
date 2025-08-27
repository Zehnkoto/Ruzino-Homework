USTC_CG_ADD_LIB(
	rzpython
	SHARED
	PRIVATE_LIBS 
		GUI glm::glm spdlog::spdlog rzconsole Python3::Module nanobind-static
	COMPILE_DEFS
		NOMINMAX=1
)

target_include_directories(rzpython PUBLIC ${NB_DIR}/include)