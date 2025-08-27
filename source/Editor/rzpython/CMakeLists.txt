USTC_CG_ADD_LIB(
	rzpython
	SHARED
	PRIVATE_LIBS 
		GUI glm::glm spdlog::spdlog rzconsole Python3::Interpreter
	COMPILE_DEFS
		NOMINMAX=1
)