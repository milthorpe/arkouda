# Makefile for Arkouda
ARKOUDA_PROJECT_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

PROJECT_NAME := arkouda
ARKOUDA_SOURCE_DIR := $(ARKOUDA_PROJECT_DIR)/src
ARKOUDA_MAIN_MODULE := arkouda_server
ARKOUDA_MAKEFILES := Makefile Makefile.paths

DEFAULT_TARGET := $(ARKOUDA_MAIN_MODULE)
.PHONY: default
default: $(DEFAULT_TARGET)

VERBOSE ?= 0

CHPL := chpl

# We need to make the HDF5 API use the 1.10.x version for compatibility between 1.10 and 1.12
CHPL_FLAGS += --ccflags="-DH5_USE_110_API"

CHPL_DEBUG_FLAGS += --print-passes

ifdef ARKOUDA_DEVELOPER
ARKOUDA_QUICK_COMPILE = true
ARKOUDA_RUNTIME_CHECKS = true
endif

ifdef ARKOUDA_QUICK_COMPILE
CHPL_FLAGS += --no-checks --no-loop-invariant-code-motion --no-fast-followers --ccflags="-O0"
else
CHPL_FLAGS += --fast
endif

ifdef ARKOUDA_RUNTIME_CHECKS
CHPL_FLAGS += --checks
endif

CHPL_FLAGS += -smemTrack=true -smemThreshold=1048576

# We have seen segfaults with cache remote at some node counts
CHPL_FLAGS += --no-cache-remote

# For configs that use a fixed heap, but still have first-touch semantics
# (gasnet-ibv-large) interleave large allocations to reduce the performance hit
# from getting progressively worse NUMA affinity due to memory reuse.
CHPL_HELP := $(shell $(CHPL) --devel --help)
ifneq (,$(findstring interleave-memory,$(CHPL_HELP)))
CHPL_FLAGS += --interleave-memory
endif

# add-path: Append custom paths for non-system software.
# Note: Darwin `ld` only supports `-rpath <path>`, not `-rpath=<paths>`.
define add-path
ifneq ("$(wildcard $(1)/lib64)","")
  INCLUDE_FLAGS += -I$(1)/include -L$(1)/lib64
  CHPL_FLAGS    += -I$(1)/include -L$(1)/lib64 --ldflags="-Wl,-rpath,$(1)/lib64"
endif
INCLUDE_FLAGS += -I$(1)/include -L$(1)/lib
CHPL_FLAGS    += -I$(1)/include -L$(1)/lib --ldflags="-Wl,-rpath,$(1)/lib"
endef
# Usage: $(eval $(call add-path,/home/user/anaconda3/envs/arkouda))
#                               ^ no space after comma
-include Makefile.paths # Add entries to this file.

ifdef ARKOUDA_ZMQ_PATH
$(eval $(call add-path,$(ARKOUDA_ZMQ_PATH)))
endif
ifdef ARKOUDA_HDF5_PATH
$(eval $(call add-path,$(ARKOUDA_HDF5_PATH)))
endif
ifdef ARKOUDA_ARROW_PATH
$(eval $(call add-path,$(ARKOUDA_ARROW_PATH)))
endif
ifdef ARKOUDA_ICONV_PATH
$(eval $(call add-path,$(ARKOUDA_ICONV_PATH)))
endif
ifdef ARKOUDA_IDN2_PATH
$(eval $(call add-path,$(ARKOUDA_IDN2_PATH)))
endif

ifndef ARKOUDA_CONFIG_FILE
ARKOUDA_CONFIG_FILE := $(ARKOUDA_PROJECT_DIR)/ServerModules.cfg
endif

CHPL_FLAGS += -lhdf5 -lhdf5_hl -lzmq -liconv -lidn2 -lparquet -larrow

ARROW_FILE_NAME += $(ARKOUDA_SOURCE_DIR)/ArrowFunctions
ARROW_CPP += $(ARROW_FILE_NAME).cpp
ARROW_H += $(ARROW_FILE_NAME).h
ARROW_O += $(ARROW_FILE_NAME).o


.PHONY: install-deps
install-deps: install-zmq install-hdf5 install-arrow install-iconv install-idn2

DEP_DIR := dep
DEP_INSTALL_DIR := $(ARKOUDA_PROJECT_DIR)/$(DEP_DIR)
DEP_BUILD_DIR := $(ARKOUDA_PROJECT_DIR)/$(DEP_DIR)/build

ZMQ_VER := 4.3.5
ZMQ_NAME_VER := zeromq-$(ZMQ_VER)
ZMQ_BUILD_DIR := $(DEP_BUILD_DIR)/$(ZMQ_NAME_VER)
ZMQ_INSTALL_DIR := $(DEP_INSTALL_DIR)/zeromq-install
ZMQ_LINK := https://github.com/zeromq/libzmq/releases/download/v$(ZMQ_VER)/$(ZMQ_NAME_VER).tar.gz
install-zmq:
	@echo "Installing ZeroMQ"
	rm -rf $(ZMQ_BUILD_DIR) $(ZMQ_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)
	cd $(DEP_BUILD_DIR) && curl -sL $(ZMQ_LINK) | tar xz
	cd $(ZMQ_BUILD_DIR) && ./configure --prefix=$(ZMQ_INSTALL_DIR) CFLAGS=-O3 CXXFLAGS=-O3 && make && make install
	rm -r $(ZMQ_BUILD_DIR)
	echo '$$(eval $$(call add-path,$(ZMQ_INSTALL_DIR)))' >> Makefile.paths

HDF5_MAJ_MIN_VER := 1.12
HDF5_VER := 1.12.1
HDF5_NAME_VER := hdf5-$(HDF5_VER)
HDF5_BUILD_DIR := $(DEP_BUILD_DIR)/$(HDF5_NAME_VER)
HDF5_INSTALL_DIR := $(DEP_INSTALL_DIR)/hdf5-install
HDF5_LINK :=  https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-$(HDF5_MAJ_MIN_VER)/$(HDF5_NAME_VER)/src/$(HDF5_NAME_VER).tar.gz
install-hdf5:
	@echo "Installing HDF5"
	rm -rf $(HDF5_BUILD_DIR) $(HDF5_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)
	cd $(DEP_BUILD_DIR) && curl -sL $(HDF5_LINK) | tar xz
	cd $(HDF5_BUILD_DIR) && ./configure --prefix=$(HDF5_INSTALL_DIR) --enable-optimization=high --enable-hl && make && make install
	rm -rf $(HDF5_BUILD_DIR)
	echo '$$(eval $$(call add-path,$(HDF5_INSTALL_DIR)))' >> Makefile.paths

ARROW_VER := 11.0.0
ARROW_NAME_VER := apache-arrow-$(ARROW_VER)
ARROW_FULL_NAME_VER := arrow-apache-arrow-$(ARROW_VER)
ARROW_BUILD_DIR := $(DEP_BUILD_DIR)/$(ARROW_FULL_NAME_VER)
ARROW_INSTALL_DIR := $(DEP_INSTALL_DIR)/arrow-install
ARROW_LINK := https://github.com/apache/arrow/archive/refs/tags/$(ARROW_NAME_VER).tar.gz
install-arrow:
	@echo "Installing Apache Arrow/Parquet"
	rm -rf $(ARROW_BUILD_DIR) $(ARROW_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)
	cd $(DEP_BUILD_DIR) && curl -sL $(ARROW_LINK) | tar xz
	cd $(ARROW_BUILD_DIR)/cpp && cmake -DARROW_DEPENDENCY_SOURCE=AUTO -DCMAKE_INSTALL_PREFIX=$(ARROW_INSTALL_DIR) -DCMAKE_BUILD_TYPE=Release -DARROW_PARQUET=ON -DARROW_WITH_SNAPPY=ON -DARROW_WITH_BROTLI=ON -DARROW_WITH_BZ2=ON -DARROW_WITH_LZ4=ON -DARROW_WITH_ZLIB=ON -DARROW_WITH_ZSTD=ON $(ARROW_OPTIONS) . && make && make install
	rm -rf $(ARROW_BUILD_DIR)
	echo '$$(eval $$(call add-path,$(ARROW_INSTALL_DIR)))' >> Makefile.paths

ICONV_VER := 1.17
ICONV_NAME_VER := libiconv-$(ICONV_VER)
ICONV_BUILD_DIR := $(DEP_BUILD_DIR)/$(ICONV_NAME_VER)
ICONV_INSTALL_DIR := $(DEP_INSTALL_DIR)/libiconv-install
ICONV_LINK := https://ftp.gnu.org/pub/gnu/libiconv/libiconv-$(ICONV_VER).tar.gz
install-iconv:
	@echo "Installing iconv"
	rm -rf $(ICONV_BUILD_DIR) $(ICONV_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)
	cd $(DEP_BUILD_DIR) && curl -sL $(ICONV_LINK) | tar xz
	cd $(ICONV_BUILD_DIR) && ./configure --prefix=$(ICONV_INSTALL_DIR) && make && make install
	rm -rf $(ICONV_BUILD_DIR)
	echo '$$(eval $$(call add-path,$(ICONV_INSTALL_DIR)))' >> Makefile.paths

LIBIDN_VER := 2.3.4
LIBIDN_NAME_VER := libidn2-$(LIBIDN_VER)
LIBIDN_BUILD_DIR := $(DEP_BUILD_DIR)/$(LIBIDN_NAME_VER)
LIBIDN_INSTALL_DIR := $(DEP_INSTALL_DIR)/libidn2-install
LIBIDN_LINK := https://ftp.gnu.org/gnu/libidn/libidn2-$(LIBIDN_VER).tar.gz
install-idn2:
	@echo "Installing libidn2"
	rm -rf $(LIBIDN_BUILD_DIR) $(LIBIDN_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)
	cd $(DEP_BUILD_DIR) && curl -sL $(LIBIDN_LINK) | tar xz
	cd $(LIBIDN_BUILD_DIR) && ./configure --prefix=$(LIBIDN_INSTALL_DIR) && make && make install
	rm -rf $(LIBIDN_BUILD_DIR)
	echo '$$(eval $$(call add-path,$(LIBIDN_INSTALL_DIR)))' >> Makefile.paths

# System Environment
ifdef LD_RUN_PATH
#CHPL_FLAGS += --ldflags="-Wl,-rpath=$(LD_RUN_PATH)"
# This pattern handles multiple paths separated by :
TEMP_FLAGS = $(patsubst %,--ldflags="-Wl+-rpath+%",$(strip $(subst :, ,$(LD_RUN_PATH))))
# The comma hack is necessary because commas can't appear in patsubst args
comma:= ,
CHPL_FLAGS += $(subst +,$(comma),$(TEMP_FLAGS))
endif

ifdef LD_LIBRARY_PATH
CHPL_FLAGS += $(patsubst %,-L%,$(strip $(subst :, ,$(LD_LIBRARY_PATH))))
endif

.PHONY: check-deps
ifndef ARKOUDA_SKIP_CHECK_DEPS
CHECK_DEPS = check-chpl check-zmq check-hdf5 check-re2 check-arrow check-iconv check-idn2
endif
check-deps: $(CHECK_DEPS)

SANITIZER = $(shell $(CHPL_HOME)/util/chplenv/chpl_sanitizers.py --exe 2>/dev/null)
ifneq ($(SANITIZER),none)
ARROW_SANITIZE=-fsanitize=$(SANITIZER)
endif

CHPL_CXX = $(shell $(CHPL_HOME)/util/config/compileline --compile-c++ 2>/dev/null)
ifeq ($(CHPL_CXX),none)
CHPL_CXX=$(CXX)
endif

.PHONY: compile-arrow-cpp
compile-arrow-cpp:
	$(CHPL_CXX) -O3 -std=c++17 -c $(ARROW_CPP) -o $(ARROW_O) $(INCLUDE_FLAGS) $(ARROW_SANITIZE)

$(ARROW_O): $(ARROW_CPP) $(ARROW_H)
	make compile-arrow-cpp

CHPL_MINOR := $(shell $(CHPL) --version | sed -n "s/chpl version 1\.\([0-9]*\).*/\1/p")
CHPL_VERSION_OK := $(shell test $(CHPL_MINOR) -ge 30 && echo yes)
CHPL_VERSION_WARN := $(shell test $(CHPL_MINOR) -le 30 && echo yes)
.PHONY: check-chpl
check-chpl:
ifneq ($(CHPL_VERSION_OK),yes)
	$(error Chapel 1.30.0 or newer is required)
endif
ifeq ($(CHPL_VERSION_WARN),yes)
	$(warning Chapel 1.31.0 or newer is recommended)
endif

ZMQ_CHECK = $(DEP_INSTALL_DIR)/checkZMQ.chpl
check-zmq: $(ZMQ_CHECK)
	@echo "Checking for ZMQ"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/master/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

HDF5_CHECK = $(DEP_INSTALL_DIR)/checkHDF5.chpl
check-hdf5: $(HDF5_CHECK)
	@echo "Checking for HDF5"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/master/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

RE2_CHECK = $(DEP_INSTALL_DIR)/checkRE2.chpl
check-re2: $(RE2_CHECK)
	@echo "Checking for RE2"
	@$(CHPL) $(CHPL_FLAGS) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/master/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

ARROW_CHECK = $(DEP_INSTALL_DIR)/checkArrow.chpl
check-arrow: $(ARROW_CHECK) $(ARROW_O)
	@echo "Checking for Arrow"
	make compile-arrow-cpp
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) $< $(ARROW_M) -M $(ARKOUDA_SOURCE_DIR) -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/master/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

ICONV_CHECK = $(DEP_INSTALL_DIR)/checkIconv.chpl
check-iconv: $(ICONV_CHECK)
	@echo "Checking for iconv"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) -M $(ARKOUDA_SOURCE_DIR) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/master/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

IDN2_CHECK = $(DEP_INSTALL_DIR)/checkIdn2.chpl
check-idn2: $(IDN2_CHECK)
	@echo "Checking for idn2"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) -M $(ARKOUDA_SOURCE_DIR) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/master/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

ALL_TARGETS := $(ARKOUDA_MAIN_MODULE)
.PHONY: all
all: $(ALL_TARGETS)

Makefile.paths:
	@touch $@

# args: RuleTarget DefinedHelpText
define create_help_target
export $(2)
HELP_TARGETS += $(1)
.PHONY: $(1)
$(1):
	@echo "$$$$$(2)"
endef

####################
#### Arkouda.mk ####
####################

define ARKOUDA_HELP_TEXT
# default		$(DEFAULT_TARGET)
  help
  all			$(ALL_TARGETS)

# $(ARKOUDA_MAIN_MODULE)	Can override CHPL_FLAGS
  arkouda-help
  arkouda-clean

  tags			Create developer TAGS file
  tags-clean

endef
$(eval $(call create_help_target,arkouda-help,ARKOUDA_HELP_TEXT))

# Set the arkouda server version from the VERSION file
VERSION=$(shell python3 -c "import versioneer; print(versioneer.get_versions()[\"version\"])")
# Test for existence of VERSION file
# ifneq ("$(wildcard $(VERSIONFILE))","")
# 	VERSION=$(shell cat ${VERSIONFILE})
# else
# 	VERSION=$(shell date +'%Y.%m.%d')
# endif

# Version needs to be escape-quoted for chpl to interpret as string
CHPL_FLAGS_WITH_VERSION = $(CHPL_FLAGS)
CHPL_FLAGS_WITH_VERSION += -sarkoudaVersion="\"$(VERSION)\""

ifdef ARKOUDA_PRINT_PASSES_FILE
	PRINT_PASSES_FLAGS := --print-passes-file $(ARKOUDA_PRINT_PASSES_FILE)
endif

ifdef REGEX_MAX_CAPTURES
	REGEX_MAX_CAPTURES_FLAG = -sregexMaxCaptures=$(REGEX_MAX_CAPTURES)
endif

ARKOUDA_SOURCES = $(shell find $(ARKOUDA_SOURCE_DIR)/ -type f -name '*.chpl')
ARKOUDA_MAIN_SOURCE := $(ARKOUDA_SOURCE_DIR)/$(ARKOUDA_MAIN_MODULE).chpl

ifeq ($(shell expr $(CHPL_MINOR) \> 31),1)
	ARKOUDA_COMPAT_MODULES += -M $(ARKOUDA_SOURCE_DIR)/compat/gt-131
endif

ifeq ($(shell expr $(CHPL_MINOR) \= 31),1)
	CHPL_COMPAT_FLAGS += -sbigintInitThrows=true
	ARKOUDA_COMPAT_MODULES += -M $(ARKOUDA_SOURCE_DIR)/compat/eq-131
endif

ifeq ($(shell expr $(CHPL_MINOR) \= 30),1)
	CHPL_COMPAT_FLAGS += -sbigintInitThrows=true
	ARKOUDA_COMPAT_MODULES += -M $(ARKOUDA_SOURCE_DIR)/compat/e-130
endif

MULTIGPUSORT_DIR?=$(HOME)/multi-gpu-sorting # this can be overridden on the command line
CUDACXX_FLAGS+=-fopenmp -I$(MULTIGPUSORT_DIR)/hipsrc

CUDA_OBJECTS=src/cubMin.cu.o src/cubMax.cu.o src/cubSum.cu.o src/cubSort.cu.o src/cubHistogram.cu.o src/ncclCollectives.cu.o src/multiGPUMergeSort.cu.o
CUDA_SOURCES=$(CUDA_OBJECTS:.cu.o=.cu)
CUDA_HEADERS=$(CUDA_SOURCES:.cu=.h)

%.cu.o:	%.cu
	hipcc -g -std=c++14 -fopenmp -I$(MULTIGPUSORT_DIR)/hipsrc -I$(HIP_ROOT_DIR)/include -c $< -o $@

ifndef CHPL_GPU_HOME
$(error CHPL_GPU_HOME not defined)
endif

GPU_FLAGS=-M $(CHPL_GPU_HOME)/modules $(CHPL_GPU_HOME)/include/GPUAPI.h $(CUDA_HEADERS) -I$(ZMQ_DIR)/include -I$(HDF5_DIR)/include -I$(ARROW_DIR)/include -L$(CHPL_GPU_HOME)/lib -L$(CHPL_GPU_HOME)/lib64 -L$(HIP_ROOT_DIR)/lib -lGPUAPIHIP_static -lamdhip64 -L$(RCCL_PATH)/lib -lrccl -L$(ROCM_PATH)/llvm/lib -lomp 

MODULE_GENERATION_SCRIPT=$(ARKOUDA_SOURCE_DIR)/serverModuleGen.py
# This is the main compilation statement section
$(ARKOUDA_MAIN_MODULE): check-deps $(ARROW_O) $(ARKOUDA_SOURCES) $(CUDA_HEADERS) $(CUDA_OBJECTS) $(ARKOUDA_MAKEFILES)
	$(eval MOD_GEN_OUT=$(shell python3 $(MODULE_GENERATION_SCRIPT) $(ARKOUDA_CONFIG_FILE) $(ARKOUDA_SOURCE_DIR)))
	$(CHPL) $(CHPL_DEBUG_FLAGS) -g --detailed-errors --ldflags -no-pie  $(PRINT_PASSES_FLAGS) $(REGEX_MAX_CAPTURES_FLAG) $(OPTIONAL_SERVER_FLAGS) $(CHPL_FLAGS_WITH_VERSION) $(CHPL_COMPAT_FLAGS) $(ARKOUDA_MAIN_SOURCE) $(CUDA_HEADERS) $(CUDA_OBJECTS) $(ARKOUDA_COMPAT_MODULES) $(ARKOUDA_SERVER_USER_MODULES) $(GPU_FLAGS) $(MOD_GEN_OUT) -o $@

CLEAN_TARGETS += arkouda-clean
.PHONY: arkouda-clean
arkouda-clean:
	$(RM) $(ARKOUDA_MAIN_MODULE) $(ARKOUDA_MAIN_MODULE)_real $(ARROW_O)

.PHONY: tags
tags:
	-@(cd $(ARKOUDA_SOURCE_DIR) && $(CHPL_HOME)/util/chpltags -r . > /dev/null \
		&& echo "Updated $(ARKOUDA_SOURCE_DIR)/TAGS" \
		|| echo "Tags utility not available.  Skipping tags generation.")

CLEANALL_TARGETS += tags-clean
.PHONY: tags-clean
tags-clean:
	$(RM) $(ARKOUDA_SOURCE_DIR)/TAGS

####################
#### Archive.mk ####
####################

define ARCHIVE_HELP_TEXT
# archive		COMMIT=$(COMMIT)
  archive-help
  archive-clean

endef
$(eval $(call create_help_target,archive-help,ARCHIVE_HELP_TEXT))

COMMIT ?= master
ARCHIVE_EXTENSION := tar.gz
ARCHIVE_FILENAME := $(PROJECT_NAME)-$(subst /,_,$(COMMIT)).$(ARCHIVE_EXTENSION)

.PHONY: archive
archive: $(ARCHIVE_FILENAME)

.PHONY: $(ARCHIVE_FILENAME)
$(ARCHIVE_FILENAME):
	git archive --format=$(ARCHIVE_EXTENSION) --prefix=$(subst .$(ARCHIVE_EXTENSION),,$(ARCHIVE_FILENAME))/ $(COMMIT) > $@

CLEANALL_TARGETS += archive-clean
.PHONY: archive-clean
archive-clean:
	$(RM) $(PROJECT_NAME)-*.$(ARCHIVE_EXTENSION)

################
#### Doc.mk ####
################

define DOC_HELP_TEXT
# doc			Generate $(DOC_DIR)/ with doc-* for server, etc.
  doc-help
  doc-clean
  doc-server
  doc-python

endef
$(eval $(call create_help_target,doc-help,DOC_HELP_TEXT))

DOC_DIR := docs
DOC_SERVER_OUTPUT_DIR := $(ARKOUDA_PROJECT_DIR)/$(DOC_DIR)/server
DOC_PYTHON_OUTPUT_DIR := $(ARKOUDA_PROJECT_DIR)/$(DOC_DIR)

DOC_COMPONENTS := \
	$(DOC_SERVER_OUTPUT_DIR) \
	$(DOC_PYTHON_OUTPUT_DIR)
$(DOC_COMPONENTS):
	mkdir -p $@

$(DOC_DIR):
	mkdir -p $@

.PHONY: doc
doc: doc-python doc-server 

CHPLDOC := chpldoc
CHPLDOC_FLAGS := --process-used-modules
.PHONY: doc-server
doc-server: ${DOC_DIR} $(DOC_SERVER_OUTPUT_DIR)/index.html
$(DOC_SERVER_OUTPUT_DIR)/index.html: $(ARKOUDA_SOURCES) $(ARKOUDA_MAKEFILES) | $(DOC_SERVER_OUTPUT_DIR)
	@echo "Building documentation for: Server"
	@# Build the documentation to the Chapel output directory
	$(CHPLDOC) $(CHPLDOC_FLAGS) $(ARKOUDA_MAIN_SOURCE) $(ARKOUDA_SOURCE_DIR)/compat/e-130/* -o $(DOC_SERVER_OUTPUT_DIR)
	@# Create the .nojekyll file needed for github pages in the  Chapel output directory
	touch $(DOC_SERVER_OUTPUT_DIR)/.nojekyll
	@echo "Completed building documentation for: Server"

DOC_PYTHON_SOURCE_DIR := pydoc
DOC_PYTHON_SOURCES = $(shell find $(DOC_PYTHON_SOURCE_DIR)/ -type f)
.PHONY: doc-python
doc-python: ${DOC_DIR} $(DOC_PYTHON_OUTPUT_DIR)/index.html
$(DOC_PYTHON_OUTPUT_DIR)/index.html: $(DOC_PYTHON_SOURCES) $(ARKOUDA_MAKEFILES)
	@echo "Building documentation for: Python"
	$(eval $@_TMP := $(shell mktemp -d))
	@# Build the documentation to a temporary output directory.
	cd $(DOC_PYTHON_SOURCE_DIR) && $(MAKE) BUILDDIR=$($@_TMP) html
	@# Delete old python docs but retain Chapel docs in $(DOC_SERVER_OUTPUT_DIR).
	$(RM) -r docs/*html docs/*js docs/_static docs/_sources docs/autoapi docs/setup/ docs/usage docs/*inv
	@# Move newly-generated python docs including .nojekyll file needed for github pages.
	mv $($@_TMP)/html/* $($@_TMP)/html/.nojekyll $(DOC_PYTHON_OUTPUT_DIR)
	@# Remove temporary directory.
	$(RM) -r $($@_TMP)
	@# Remove server/index.html placeholder file to prepare for doc-server content
	$(RM) docs/server/index.html
	@echo "Completed building documentation for: Python"

CLEAN_TARGETS += doc-clean
.PHONY: doc-clean
doc-clean:
	$(RM) -r $(DOC_DIR)

check:
	@$(ARKOUDA_PROJECT_DIR)/server_util/test/checkInstall

#################
#### Test.mk ####
#################

TEST_SOURCE_DIR := tests/server
TEST_SOURCES := $(wildcard $(TEST_SOURCE_DIR)/*.chpl)
TEST_MODULES := $(basename $(notdir $(TEST_SOURCES)))

TEST_BINARY_DIR := test-bin
TEST_BINARY_SIGIL := #t-
TEST_TARGETS := $(addprefix $(TEST_BINARY_DIR)/$(TEST_BINARY_SIGIL),$(TEST_MODULES))

ifeq ($(VERBOSE),1)
TEST_CHPL_FLAGS ?= $(CHPL_DEBUG_FLAGS) $(CHPL_FLAGS)
else
TEST_CHPL_FLAGS ?= $(CHPL_FLAGS)
endif

define TEST_HELP_TEXT
# test			Build all tests ($(TEST_BINARY_DIR)/$(TEST_BINARY_SIGIL)*); Can override TEST_CHPL_FLAGS
  test-help
  test-clean
 $(foreach t,$(sort $(TEST_TARGETS)), $(t)\n)
endef
$(eval $(call create_help_target,test-help,TEST_HELP_TEXT))

.PHONY: test
test: test-python

.PHONY: test-chapel
test-chapel:
	start_test $(TEST_SOURCE_DIR)

.PHONY: test-all
test-all: test-python test-chapel

mypy:
	python3 -m mypy arkouda

$(TEST_BINARY_DIR):
	mkdir -p $(TEST_BINARY_DIR)

.PHONY: $(TEST_TARGETS) # Force tests to always rebuild.
$(TEST_TARGETS): $(TEST_BINARY_DIR)/$(TEST_BINARY_SIGIL)%: $(TEST_SOURCE_DIR)/%.chpl | $(TEST_BINARY_DIR)
	$(CHPL) $(TEST_CHPL_FLAGS) -M $(ARKOUDA_SOURCE_DIR) $(ARKOUDA_COMPAT_MODULES) $< -o $@

print-%:
	$(info $($*)) @true

test-python: 
	python3 -m pytest $(ARKOUDA_PYTEST_OPTIONS) -c pytest.ini

CLEAN_TARGETS += test-clean
.PHONY: test-clean
test-clean:
	$(RM) $(TEST_TARGETS) $(addsuffix _real,$(TEST_TARGETS))

.PHONY: benchmark
benchmark:
	python3 -m pytest -c benchmark.ini --benchmark-autosave --benchmark-storage=file://benchmark_v2/.benchmarks

version:
	@echo $(VERSION);

#####################
#### Epilogue.mk ####
#####################

define CLEAN_HELP_TEXT
# clean
  clean-help
 $(foreach t,$(CLEAN_TARGETS), $(t)\n)
endef
$(eval $(call create_help_target,clean-help,CLEAN_HELP_TEXT))

.PHONY: clean cleanall
clean: $(CLEAN_TARGETS)
cleanall: clean $(CLEANALL_TARGETS)

.PHONY: help
help: $(HELP_TARGETS)

